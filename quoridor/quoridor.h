#pragma once

// Quoridor engine, C++ port of the Python reference implementation in
// alphazero/quoridor.py.  The semantics are IDENTICAL by design — same
// canonical frame (observations and actions from the perspective of the
// player to move, who always travels toward row 0), same action encoding
// (12 pawn moves, then (N-1)^2 horizontal and (N-1)^2 vertical wall
// slots), same step()/reward/truncation behaviour — and the equivalence
// is enforced by lockstep cross-validation tests on the Python side
// (alphazero/test_cpp_backend.py).  Change the two files together.

#include <string>
#include <utility>
#include <vector>

namespace quoridor {

constexpr int NUM_PAWN_ACTIONS = 12;

// Canonical deltas for actions 0-11: N,S,E,W, NN,SS,EE,WW, NE,NW,SE,SW.
constexpr int PAWN_DELTAS[NUM_PAWN_ACTIONS][2] = {
    {-1, 0}, {1, 0}, {0, 1}, {0, -1},
    {-2, 0}, {2, 0}, {0, 2}, {0, -2},
    {-1, 1}, {-1, -1}, {1, 1}, {1, -1},
};

inline int num_actions(int n) { return NUM_PAWN_ACTIONS + 2 * (n - 1) * (n - 1); }
inline int obs_size(int n) { return 2 * n * n + 2 * (n - 1) * (n - 1) + 2; }

struct StepOutcome {
    double reward;
    bool terminated;
    bool truncated;
};

class Engine {
public:
    explicit Engine(int board_size = 5, int walls = 3, int max_moves = 0);

    void reset();
    StepOutcome step(int action);

    // Legal actions for the player to move, canonical frame, ascending.
    const std::vector<int>& legal_actions() const;
    std::vector<double> observation() const;
    std::string to_string() const;

    // Install an arbitrary position (testing / analysis).
    void set_state(std::pair<int, int> p1, std::pair<int, int> p2,
                   int to_play,
                   const std::vector<std::pair<int, int>>& h_walls,
                   const std::vector<std::pair<int, int>>& v_walls,
                   int walls_p1, int walls_p2);

    // Absolute-coordinate helpers (used by the CLI): the matching canonical
    // action, or -1 when the move/wall is not currently legal.
    int action_for_move(int r, int c) const;
    int action_for_wall(int r, int c, char orientation) const;

    int board_size() const { return n_; }
    int max_walls() const { return max_walls_; }
    int max_moves() const { return max_moves_; }
    int action_count() const { return num_actions_; }
    int obs_dim() const { return obs_dim_; }
    int to_play() const { return to_play_; }
    bool done() const { return done_; }
    int winner() const { return winner_; }  // meaningful only when done()
    int move_count() const { return move_count_; }
    int walls_left(int player) const { return walls_left_[idx(player)]; }
    std::pair<int, int> pawn(int player) const {
        return {pos_[idx(player)][0], pos_[idx(player)][1]};
    }
    std::vector<std::pair<int, int>> h_walls() const;
    std::vector<std::pair<int, int>> v_walls() const;

private:
    static int idx(int player) { return player == 1 ? 0 : 1; }
    int can_wall_row(int r) const { return to_play_ == 1 ? r : n_ - 2 - r; }

    // Bounds-safe wall queries in the CANONICAL frame of the player to
    // move (out-of-range slots read as empty, like Python set lookups).
    bool hwall_c(int r, int c) const;
    bool vwall_c(int r, int c) const;

    std::pair<int, int> canonical_pos(int player) const;
    bool step_blocked(int r, int c, int dr, int dc) const;
    bool path_exists(std::pair<int, int> start, int goal_row) const;
    bool wall_ok(char orientation, int r, int c,
                 const std::vector<char>& nodes,
                 std::pair<int, int> own, std::pair<int, int> opp) const;
    void compute_legal() const;
    void invalidate() const { cache_valid_ = false; }

    int n_, max_walls_, max_moves_, num_actions_, obs_dim_;
    int pos_[2][2];       // absolute pawn positions, [0] = player +1
    int walls_left_[2];
    // Absolute (n-1)x(n-1) wall grids; mutable because wall legality
    // temporarily places a candidate wall for the path probe (always
    // restored before returning).
    mutable std::vector<char> hgrid_, vgrid_;
    int to_play_;         // +1 / -1
    bool done_;
    int winner_;          // +1 / -1 / 0 (draw); valid only when done_
    int move_count_;

    mutable std::vector<int> legal_cache_;
    mutable std::vector<char> legal_mask_;
    mutable bool cache_valid_;
};

}  // namespace quoridor
