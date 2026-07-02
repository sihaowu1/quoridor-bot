#include "quoridor.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace quoridor {

Engine::Engine(int board_size, int walls, int max_moves)
    : n_(board_size),
      max_walls_(walls),
      max_moves_(max_moves > 0 ? max_moves
                               : 12 * board_size * board_size / 5),
      num_actions_(quoridor::num_actions(board_size)),
      obs_dim_(quoridor::obs_size(board_size)) {
    if (board_size < 3) {
        throw std::invalid_argument("board_size must be at least 3");
    }
    hgrid_.assign((n_ - 1) * (n_ - 1), 0);
    vgrid_.assign((n_ - 1) * (n_ - 1), 0);
    reset();
}

void Engine::reset() {
    pos_[0][0] = n_ - 1;
    pos_[0][1] = n_ / 2;
    pos_[1][0] = 0;
    pos_[1][1] = n_ / 2;
    walls_left_[0] = walls_left_[1] = max_walls_;
    std::fill(hgrid_.begin(), hgrid_.end(), 0);
    std::fill(vgrid_.begin(), vgrid_.end(), 0);
    to_play_ = 1;
    done_ = false;
    winner_ = 0;
    move_count_ = 0;
    invalidate();
}

bool Engine::hwall_c(int r, int c) const {
    if (r < 0 || c < 0 || r >= n_ - 1 || c >= n_ - 1) return false;
    return hgrid_[(to_play_ == 1 ? r : n_ - 2 - r) * (n_ - 1) + c] != 0;
}

bool Engine::vwall_c(int r, int c) const {
    if (r < 0 || c < 0 || r >= n_ - 1 || c >= n_ - 1) return false;
    return vgrid_[(to_play_ == 1 ? r : n_ - 2 - r) * (n_ - 1) + c] != 0;
}

std::pair<int, int> Engine::canonical_pos(int player) const {
    int r = pos_[idx(player)][0];
    int c = pos_[idx(player)][1];
    return to_play_ == 1 ? std::make_pair(r, c)
                         : std::make_pair(n_ - 1 - r, c);
}

// Wall between (r, c) and (r+dr, c+dc)?  Single orthogonal steps only,
// canonical frame.
bool Engine::step_blocked(int r, int c, int dr, int dc) const {
    if (dr == -1) return hwall_c(r - 1, c) || hwall_c(r - 1, c - 1);
    if (dr == 1) return hwall_c(r, c) || hwall_c(r, c - 1);
    if (dc == 1) return vwall_c(r, c) || vwall_c(r - 1, c);
    return vwall_c(r, c - 1) || vwall_c(r - 1, c - 1);
}

bool Engine::path_exists(std::pair<int, int> start, int goal_row) const {
    static const int DIRS[4][2] = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
    std::vector<char> seen(n_ * n_, 0);
    std::vector<int> stack;
    seen[start.first * n_ + start.second] = 1;
    stack.push_back(start.first * n_ + start.second);
    while (!stack.empty()) {
        int cell = stack.back();
        stack.pop_back();
        int r = cell / n_, c = cell % n_;
        if (r == goal_row) return true;
        for (const auto& d : DIRS) {
            int nr = r + d[0], nc = c + d[1];
            if (nr < 0 || nr >= n_ || nc < 0 || nc >= n_) continue;
            if (seen[nr * n_ + nc]) continue;
            if (step_blocked(r, c, d[0], d[1])) continue;
            seen[nr * n_ + nc] = 1;
            stack.push_back(nr * n_ + nc);
        }
    }
    return false;
}

bool Engine::wall_ok(char orientation, int r, int c,
                     const std::vector<char>& nodes,
                     std::pair<int, int> own,
                     std::pair<int, int> opp) const {
    int m = n_ - 1;
    int e1r, e1c, midr, midc, e2r, e2c;
    bool e1_border, e2_border;
    if (orientation == 'H') {
        if (hwall_c(r, c) || vwall_c(r, c)) return false;       // slot / cross
        if (hwall_c(r, c - 1) || hwall_c(r, c + 1)) return false;  // overlap
        e1r = r + 1; e1c = c;
        midr = r + 1; midc = c + 1;
        e2r = r + 1; e2c = c + 2;
        e1_border = c == 0;
        e2_border = c + 2 == n_;
    } else {
        if (vwall_c(r, c) || hwall_c(r, c)) return false;
        if (vwall_c(r - 1, c) || vwall_c(r + 1, c)) return false;
        e1r = r; e1c = c + 1;
        midr = r + 1; midc = c + 1;
        e2r = r + 2; e2c = c + 1;
        e1_border = r == 0;
        e2_border = r + 2 == n_;
    }

    // A wall can only cut off a path if it is part of a barrier connecting
    // two border points, which requires it to hook into the border /
    // existing walls at TWO of its three lattice nodes.
    int stride = n_ + 1;
    int anchors = ((e1_border || nodes[e1r * stride + e1c]) ? 1 : 0) +
                  ((e2_border || nodes[e2r * stride + e2c]) ? 1 : 0) +
                  (nodes[midr * stride + midc] ? 1 : 0);
    if (anchors < 2) return true;

    // Hypothetically place the wall (absolute slot) and probe both paths.
    int ar = can_wall_row(r);
    std::vector<char>& grid = orientation == 'H' ? hgrid_ : vgrid_;
    grid[ar * m + c] = 1;
    bool ok = path_exists(own, 0) && path_exists(opp, n_ - 1);
    grid[ar * m + c] = 0;
    return ok;
}

void Engine::compute_legal() const {
    legal_cache_.clear();
    legal_mask_.assign(num_actions_, 0);
    cache_valid_ = true;
    if (done_) return;

    auto own = canonical_pos(to_play_);
    auto opp = canonical_pos(-to_play_);

    for (int a = 0; a < 4; a++) {
        int dr = PAWN_DELTAS[a][0], dc = PAWN_DELTAS[a][1];
        int nr = own.first + dr, nc = own.second + dc;
        if (nr < 0 || nr >= n_ || nc < 0 || nc >= n_) continue;
        if (step_blocked(own.first, own.second, dr, dc)) continue;
        if (nr != opp.first || nc != opp.second) {
            legal_cache_.push_back(a);
            continue;
        }
        // The opponent stands on the destination: jump rules.
        int jr = nr + dr, jc = nc + dc;
        if (jr >= 0 && jr < n_ && jc >= 0 && jc < n_ &&
            !step_blocked(nr, nc, dr, dc)) {
            legal_cache_.push_back(4 + a);  // straight jump
        } else {
            // Straight jump blocked by a wall or the board edge: the two
            // squares diagonally beside the opponent become reachable.
            int perp[2][2];
            if (dc == 0) {
                perp[0][0] = 0; perp[0][1] = 1;
                perp[1][0] = 0; perp[1][1] = -1;
            } else {
                perp[0][0] = 1; perp[0][1] = 0;
                perp[1][0] = -1; perp[1][1] = 0;
            }
            for (const auto& pd : perp) {
                int tr = nr + pd[0], tc = nc + pd[1];
                if (tr < 0 || tr >= n_ || tc < 0 || tc >= n_) continue;
                if (step_blocked(nr, nc, pd[0], pd[1])) continue;
                int ddr = dr + pd[0], ddc = dc + pd[1];
                int diag = ddr == -1 ? (ddc == 1 ? 8 : 9)
                                     : (ddc == 1 ? 10 : 11);
                legal_cache_.push_back(diag);
            }
        }
    }

    if (walls_left_[idx(to_play_)] > 0) {
        int m = n_ - 1;
        int stride = n_ + 1;
        // Lattice nodes covered by existing walls (canonical frame), used
        // by wall_ok to skip path probes for walls that cannot possibly
        // complete a barrier.
        std::vector<char> nodes(stride * stride, 0);
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < m; c++) {
                if (hwall_c(r, c)) {
                    nodes[(r + 1) * stride + c] = 1;
                    nodes[(r + 1) * stride + c + 1] = 1;
                    nodes[(r + 1) * stride + c + 2] = 1;
                }
                if (vwall_c(r, c)) {
                    nodes[r * stride + c + 1] = 1;
                    nodes[(r + 1) * stride + c + 1] = 1;
                    nodes[(r + 2) * stride + c + 1] = 1;
                }
            }
        }
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < m; c++) {
                if (wall_ok('H', r, c, nodes, own, opp)) {
                    legal_cache_.push_back(NUM_PAWN_ACTIONS + r * m + c);
                }
                if (wall_ok('V', r, c, nodes, own, opp)) {
                    legal_cache_.push_back(NUM_PAWN_ACTIONS + m * m + r * m + c);
                }
            }
        }
    }

    std::sort(legal_cache_.begin(), legal_cache_.end());
    for (int a : legal_cache_) legal_mask_[a] = 1;
}

const std::vector<int>& Engine::legal_actions() const {
    if (!cache_valid_) compute_legal();
    return legal_cache_;
}

StepOutcome Engine::step(int action) {
    if (done_) {
        throw std::runtime_error(
            "step() called on a finished game; call reset()");
    }
    if (action < 0 || action >= num_actions_) {
        throw std::invalid_argument("action must be in [0, " +
                                    std::to_string(num_actions_) + "), got " +
                                    std::to_string(action));
    }

    int mover = to_play_;

    legal_actions();  // ensure the mask is populated
    if (!legal_mask_[action]) {
        // Illegal move: immediate loss for the mover.
        done_ = true;
        winner_ = -mover;
        invalidate();
        return {-1.0, true, false};
    }

    bool won = false;
    if (action < NUM_PAWN_ACTIONS) {
        auto own = canonical_pos(mover);
        int cr = own.first + PAWN_DELTAS[action][0];   // canonical dest
        int cc = own.second + PAWN_DELTAS[action][1];
        won = cr == 0;                                 // canonical goal row
        pos_[idx(mover)][0] = mover == 1 ? cr : n_ - 1 - cr;
        pos_[idx(mover)][1] = cc;
    } else {
        int m = n_ - 1;
        int w = action - NUM_PAWN_ACTIONS;
        bool horizontal = w < m * m;
        if (!horizontal) w -= m * m;
        int ar = can_wall_row(w / m);
        (horizontal ? hgrid_ : vgrid_)[ar * m + w % m] = 1;
        walls_left_[idx(mover)]--;
    }

    move_count_++;
    invalidate();

    bool truncated = false;
    double reward = 0.0;
    if (won) {
        done_ = true;
        winner_ = mover;
        reward = 1.0;
    }

    to_play_ = -mover;

    if (!done_ && move_count_ >= max_moves_) {
        done_ = true;
        winner_ = 0;
        truncated = true;
    }
    // Safety net (mirrors the Python env): a player with no move at all
    // ends the game as a draw rather than crashing the search.
    if (!done_ && legal_actions().empty()) {
        done_ = true;
        winner_ = 0;
        truncated = true;
    }

    return {reward, done_ && !truncated, truncated};
}

std::vector<double> Engine::observation() const {
    std::vector<double> obs(obs_dim_, 0.0);
    auto own = canonical_pos(to_play_);
    auto opp = canonical_pos(-to_play_);
    int m = n_ - 1;
    obs[own.first * n_ + own.second] = 1.0;
    obs[n_ * n_ + opp.first * n_ + opp.second] = 1.0;
    int base = 2 * n_ * n_;
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < m; c++) {
            if (hwall_c(r, c)) obs[base + r * m + c] = 1.0;
            if (vwall_c(r, c)) obs[base + m * m + r * m + c] = 1.0;
        }
    }
    double denom = std::max(max_walls_, 1);
    obs[obs_dim_ - 2] = walls_left_[idx(to_play_)] / denom;
    obs[obs_dim_ - 1] = walls_left_[idx(-to_play_)] / denom;
    return obs;
}

void Engine::set_state(std::pair<int, int> p1, std::pair<int, int> p2,
                       int to_play,
                       const std::vector<std::pair<int, int>>& h_walls,
                       const std::vector<std::pair<int, int>>& v_walls,
                       int walls_p1, int walls_p2) {
    auto in_board = [&](std::pair<int, int> p) {
        return p.first >= 0 && p.first < n_ && p.second >= 0 && p.second < n_;
    };
    if (!in_board(p1) || !in_board(p2) || p1 == p2) {
        throw std::invalid_argument("invalid pawn positions");
    }
    if (to_play != 1 && to_play != -1) {
        throw std::invalid_argument("to_play must be +1 or -1");
    }
    std::fill(hgrid_.begin(), hgrid_.end(), 0);
    std::fill(vgrid_.begin(), vgrid_.end(), 0);
    int m = n_ - 1;
    for (const auto& [walls, grid] :
         {std::make_pair(&h_walls, &hgrid_), std::make_pair(&v_walls, &vgrid_)}) {
        for (const auto& [r, c] : *walls) {
            if (r < 0 || r >= m || c < 0 || c >= m) {
                throw std::invalid_argument("wall slot out of range");
            }
            (*grid)[r * m + c] = 1;
        }
    }
    pos_[0][0] = p1.first;
    pos_[0][1] = p1.second;
    pos_[1][0] = p2.first;
    pos_[1][1] = p2.second;
    to_play_ = to_play;
    walls_left_[0] = walls_p1;
    walls_left_[1] = walls_p2;
    done_ = false;
    winner_ = 0;
    move_count_ = 0;
    invalidate();
}

int Engine::action_for_move(int r, int c) const {
    if (done_) return -1;
    auto own = canonical_pos(to_play_);
    int cr = to_play_ == 1 ? r : n_ - 1 - r;  // absolute -> canonical
    legal_actions();
    for (int a = 0; a < NUM_PAWN_ACTIONS; a++) {
        if (!legal_mask_[a]) continue;
        if (own.first + PAWN_DELTAS[a][0] == cr &&
            own.second + PAWN_DELTAS[a][1] == c) {
            return a;
        }
    }
    return -1;
}

int Engine::action_for_wall(int r, int c, char orientation) const {
    if (done_) return -1;
    int m = n_ - 1;
    if (r < 0 || r >= m || c < 0 || c >= m) return -1;
    if (orientation != 'H' && orientation != 'V') return -1;
    int cr = to_play_ == 1 ? r : n_ - 2 - r;  // absolute -> canonical
    int action = NUM_PAWN_ACTIONS + (orientation == 'V' ? m * m : 0) + cr * m + c;
    legal_actions();
    return legal_mask_[action] ? action : -1;
}

std::vector<std::pair<int, int>> Engine::h_walls() const {
    std::vector<std::pair<int, int>> out;
    int m = n_ - 1;
    for (int r = 0; r < m; r++)
        for (int c = 0; c < m; c++)
            if (hgrid_[r * m + c]) out.emplace_back(r, c);
    return out;
}

std::vector<std::pair<int, int>> Engine::v_walls() const {
    std::vector<std::pair<int, int>> out;
    int m = n_ - 1;
    for (int r = 0; r < m; r++)
        for (int c = 0; c < m; c++)
            if (vgrid_[r * m + c]) out.emplace_back(r, c);
    return out;
}

std::string Engine::to_string() const {
    int m = n_ - 1;
    auto hwall_abs = [&](int r, int c) {
        return r >= 0 && c >= 0 && r < m && c < m && hgrid_[r * m + c];
    };
    auto vwall_abs = [&](int r, int c) {
        return r >= 0 && c >= 0 && r < m && c < m && vgrid_[r * m + c];
    };
    std::ostringstream out;
    for (int r = 0; r < n_; r++) {
        for (int c = 0; c < n_; c++) {
            char ch = '.';
            if (pos_[0][0] == r && pos_[0][1] == c) ch = 'X';
            if (pos_[1][0] == r && pos_[1][1] == c) ch = 'O';
            out << ch;
            if (c < n_ - 1) {
                out << (vwall_abs(r, c) || vwall_abs(r - 1, c) ? '|' : ' ');
            }
        }
        out << '\n';
        if (r < n_ - 1) {
            for (int c = 0; c < n_; c++) {
                out << (hwall_abs(r, c) || hwall_abs(r, c - 1) ? '-' : ' ');
                if (c < n_ - 1) out << ' ';
            }
            out << '\n';
        }
    }
    out << "walls left: X=" << walls_left_[0] << " O=" << walls_left_[1]
        << "  to move: " << (to_play_ == 1 ? 'X' : 'O');
    return out.str();
}

}  // namespace quoridor
