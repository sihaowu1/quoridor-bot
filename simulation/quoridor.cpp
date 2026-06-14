#include "quoridor.h"

Quoridor::Quoridor() {
    players[0] = {8, 4}; 
    players[1] = {0, 4}; 
    walls_left[0] = walls_left[1] = 10; 
    
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            for (int d = 0; d < 4; d++) {
                blocked_movement[i][j][d] = false; 
            }
        }
    }
}

bool Quoridor::in_bounds(int r, int c) {
    return (r >= 0 && r < BOARD_SIZE) && (c >= 0 && c < BOARD_SIZE); 
}

void Quoridor::add_blocked_movement(int r1, int c1, int r2, int c2) {
    int dir = -1; 

    // Figure out direction from (r1, c1) to (r2, c2) 
    if (r2 == r1 - 1 && c2 == c1) dir = 0; // Up 
    if (r2 == r1 && c2 == c1 - 1) dir = 1; // Right 
    if (r2 == r1 + 1 && c2 == c1) dir = 2; // Down 
    if (r2 == r1 && c2 == c1 + 1) dir = 3; // Left 

    if (dir == -1) return; 

    blocked_movement[r1][c1][dir] = true; 
    blocked_movement[r2][c2][(dir + 2) % 4] = true; 
}

void Quoridor::remove_blocked_movement(int r1, int c1, int r2, int c2) {
    int dir = -1;

    if (r2 == r1 - 1 && c2 == c1) dir = 0;
    if (r2 == r1 && c2 == c1 + 1) dir = 1;
    if (r2 == r1 + 1 && c2 == c1) dir = 2;
    if (r2 == r1 && c2 == c1 - 1) dir = 3;

    if (dir == -1) return;

    blocked_movement[r1][c1][dir] = false;
    blocked_movement[r2][c2][(dir + 2) % 4] = false;
}

bool Quoridor::can_move_between(int r, int c, int n_r, int n_c) {
    if (!in_bounds(n_r, n_c)) return false; 

    for (int d = 0; d < 4; d++) {
        if (r + dr[d] == n_r && c + dc[d] == n_c) {
            return !blocked_movement[r][c][d]; 
        }
    } 

    return false; 
}

bool Quoridor::has_path(int player) {
    std::queue<Position> q;
    bool seen[BOARD_SIZE][BOARD_SIZE] = {}; 

    q.push(players[player]); 

    seen[players[player].r][players[player].c] = true; 

    while (!q.empty()) {
        Position curr = q.front(); 
        q.pop(); 

        if (player == 0 && curr.r == 0) return true; 
        if (player == 1 && curr.c == 8) return true; 

        for (int d = 0; d < 4; d++) {
            int n_r = curr.r + dr[d]; 
            int n_c = curr.c + dc[d]; 

            if (!in_bounds(n_r, n_c)) continue; 
            if (seen[n_r][n_c]) continue; 
            if (blocked_movement[curr.r][curr.c][d]) continue; 

            seen[n_r][n_c] = true; 
            q.push({n_r, n_c}); 
        }
    }

    return false; 
}

bool Quoridor::move_pawn(int player, int n_r, int n_c) {
    Position curr = players[player]; 

    if (!in_bounds(n_r, n_c)) return false; 
    if (!can_move_between(curr.r, curr.c, n_r, n_c)) return false; 

    int otherPlayer = 1 - player; 
    Position other = players[otherPlayer]; 

    if (other.r == n_r && other.c == n_c) return false; 

    if (abs(n_r - curr.r) > 1 || abs(n_c - curr.c) > 1) return false; 

    players[player] = {n_r, n_c}; 

    return true; 
}

bool Quoridor::place_wall(int player, int r, int c, char orientation) {
    if (walls_left[player] <= 0) return false; 
    if (r < 0 || r >= BOARD_SIZE - 1 || c < 0 || c >= BOARD_SIZE - 1) return false; 

    auto key = std::make_tuple(r, c, orientation); 

    if (wall_positions.count(key)) return false; 

    if (orientation == 'H') {
        if (wall_positions.count({r, c - 1, 'H'}) ||
            wall_positions.count({r - 1, c, 'H'})) {
            return false; 
        }

        add_blocked_movement(r, c, r + 1, c); 
        add_blocked_movement(r, c + 1, r + 1, c + 1); 
    } else if (orientation == 'V') {
        if (wall_positions.count({r - 1, c, 'V'}) ||
            wall_positions.count({r + 1, c, 'V'})) {
            return false; 
        } 

        add_blocked_movement(r, c, r, c + 1); 
        add_blocked_movement(r + 1, c, r + 1, c + 1); 
    } 

    wall_positions.insert(key); 

    bool legal_wall = has_path(0) && has_path(1); 

    if (!legal_wall) {
        wall_positions.erase(key);

        if (orientation == 'H') {
            remove_blocked_movement(r, c, r + 1, c);
            remove_blocked_movement(r, c + 1, r + 1, c + 1);
        } else {
            remove_blocked_movement(r, c, r, c + 1);
            remove_blocked_movement(r + 1, c, r + 1, c + 1);
        }

        return false;
    }

    walls_left[player]--; 
    
    return true; 
}

bool Quoridor::winner(int player) {
    if (player == 0) {
        return players[player].r == 0;
    }

    return players[player].r == BOARD_SIZE - 1;
}

void Quoridor::print_board() {
    std::cout << "\n";

    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            char ch = '.';

            if (players[0].r == r && players[0].c == c) ch = 'A';
            if (players[1].r == r && players[1].c == c) ch = 'B';

            std::cout << ch;

            if (c < BOARD_SIZE - 1) {
                std::cout << (blocked_movement[r][c][1] ? "|" : " ");
            }
        }

        std::cout << "\n";

        if (r < BOARD_SIZE - 1) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                std::cout << (blocked_movement[r][c][2] ? "-" : " ");

                if (c < BOARD_SIZE - 1) {
                    std::cout << " ";
                }
            }

            std::cout << "\n";
        }
    }

    std::cout << "Walls left: A=" << walls_left[0]
              << " B=" << walls_left[1] << "\n";
}