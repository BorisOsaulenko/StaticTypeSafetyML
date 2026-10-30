#ifndef ML_TOKENIZER_H
#define ML_TOKENIZER_H

#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

class Tokenizer {
  using id = unsigned long long;
  using json = nlohmann::json;

public:
  explicit Tokenizer(const unsigned long long vocabulary_target)
      : vocabulary_target(vocabulary_target) {}

  void fit(const std::vector<std::string> &texts) {
    std::vector<std::vector<id>> tokens = initial_mapping_setup(texts);
    if (current_id >= vocabulary_target)
      throw std::runtime_error(
          "Initial mapping exceeds the target vocabulary size.");
    while (current_id < vocabulary_target) {
      bpe_iteration(tokens);
    }
  }

  void register_special_tokens(const std::vector<std::string> &tokens) {
    for (const auto &token : tokens) {
      special_tokens[token] = current_id++;
    }
  }

  [[nodiscard]] std::vector<id> tokenize(const std::string &text) const {
    std::vector<id> result;
    size_t pos = 0;

    while (pos < text.size()) {
      // Find the earliest occurrence of any special token starting at/after pos
      size_t best_pos = std::string::npos;
      std::string matched_token;
      for (const auto &tok : special_tokens | std::views::keys) {
        if (const size_t found = text.find(tok, pos);
            found != std::string::npos &&
            (best_pos == std::string::npos || found < best_pos)) {
          best_pos = found;
          matched_token = tok;
        }
      }

      if (best_pos == std::string::npos) {
        // No more special tokens ahead; encode the remainder as plain text
        auto plain = encode_segment(text.substr(pos));
        result.insert(result.end(), plain.begin(), plain.end());
        break;
      }

      // Encode the plain-text chunk before the special token, if any
      if (best_pos > pos) {
        auto plain = encode_segment(text.substr(pos, best_pos - pos));
        result.insert(result.end(), plain.begin(), plain.end());
      }

      // Append the special token's reserved id directly (no merging)
      result.push_back(special_tokens.at(matched_token));
      pos = best_pos + matched_token.size();
    }

    return result;
  }

  void save(const std::string &path) const {
    json j;
    to_json(j, *this);
    std::ofstream file(path);
    if (!file.is_open())
      throw std::runtime_error("Could not open file for writing: " + path);
    file << j.dump(2);
  }

  void load(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open())
      throw std::runtime_error("Could not open file for reading: " + path);
    json j;
    file >> j;
    from_json(j, *this);
  }

  friend void to_json(json &j, const Tokenizer &t) {
    j = json{{"initial_mapping", t.initial_mapping},
             {"merged_mapping", t.merged_mapping},
             {"special_tokens", t.special_tokens}};
  }

  friend void from_json(const json &j, Tokenizer &t) {
    j.at("initial_mapping").get_to(t.initial_mapping);
    j.at("merged_mapping").get_to(t.merged_mapping);
    j.at("special_tokens").get_to(t.special_tokens);
  }

private:
  struct merged_map {
    id merged_id;
    std::pair<id, id> original_ids;

    merged_map(const id merged_id, const std::pair<id, id> &original_ids)
        : merged_id(merged_id), original_ids(original_ids) {}

    friend void to_json(json &j, const merged_map &m) {
      j = json{{"merged_id", m.merged_id}, {"original_ids", m.original_ids}};
    }

    friend void from_json(const json &j, merged_map &m) {
      j.at("merged_id").get_to(m.merged_id);
      j.at("original_ids").get_to(m.original_ids);
    }
  };

  std::map<unsigned char, id> initial_mapping;
  std::map<std::string, id> special_tokens;
  std::vector<merged_map> merged_mapping;
  id current_id = 0;
  unsigned long long vocabulary_target;

  [[nodiscard]]
  std::vector<std::vector<id>>
  initial_mapping_setup(const std::vector<std::string> &texts) {
    std::vector<std::vector<id>> result(texts.size());
    for (size_t idx = 0; idx < texts.size(); ++idx) {
      for (unsigned char c : texts[idx]) {
        if (!initial_mapping.contains(c))
          initial_mapping[c] = current_id++;
        result[idx].push_back(initial_mapping[c]);
      }
    }
    return result;
  }

  void bpe_iteration(std::vector<std::vector<id>> &tokens) {
    std::map<std::pair<id, id>, unsigned long long> pair_counts;
    for (const auto &token_list : tokens) {
      for (size_t i = 0; i + 1 < token_list.size(); ++i) {
        pair_counts[{token_list[i], token_list[i + 1]}]++;
      }
    }

    const auto most_frequent_pair =
        std::ranges::max_element(pair_counts, [](const auto &a, const auto &b) {
          return a.second < b.second;
        });

    if (most_frequent_pair == pair_counts.end())
      throw std::runtime_error("No pairs found for merging.");

    id new_id = current_id++;
    merged_mapping.emplace_back(new_id, most_frequent_pair->first);

    for (auto &token_list : tokens) {
      std::vector<id> new_token_list;
      for (size_t i = 0; i < token_list.size(); ++i) {
        if (i < token_list.size() - 1 &&
            token_list[i] == most_frequent_pair->first.first &&
            token_list[i + 1] == most_frequent_pair->first.second) {
          new_token_list.push_back(new_id);
          ++i; // Skip the next token as it's merged
        } else {
          new_token_list.push_back(token_list[i]);
        }
      }
      token_list = std::move(new_token_list);
    }
  }

  [[nodiscard]] std::vector<id>
  encode_segment(const std::string &segment) const {
    std::vector<id> tokens;
    tokens.reserve(segment.size());
    for (unsigned char c : segment) {
      auto it = initial_mapping.find(c);
      if (it == initial_mapping.end())
        throw std::runtime_error(
            "Character not seen during training: byte value " +
            std::to_string(c));
      tokens.push_back(it->second);
    }

    // Replay merges in the exact order they were learned
    for (const auto &m : merged_mapping) {
      std::vector<id> merged;
      merged.reserve(tokens.size());
      for (size_t i = 0; i < tokens.size();) {
        if (i + 1 < tokens.size() && tokens[i] == m.original_ids.first &&
            tokens[i + 1] == m.original_ids.second) {
          merged.push_back(m.merged_id);
          i += 2;
        } else {
          merged.push_back(tokens[i]);
          i += 1;
        }
      }
      tokens = std::move(merged);
    }

    return tokens;
  }
};

#endif // ML_TOKENIZER_H
