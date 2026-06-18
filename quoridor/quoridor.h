#pragma once

#include <set>
#include <tuple>
#include <queue>
#include <iostream>
#include <cctype> 

const int BOARD_SIZE = 9; 

struct Position {
    int r, c; 
}; 

class Quoridor {
    private: 
        Position players[2]; 
        int walls_left[2]; 

        std::set<std::tuple<int, int, char>> wall_positions; 

        bool blocked_movement[BOARD_SIZE][BOARD_SIZE][4]; 

        int dr[4] = {-1, 0, 1, 0}; 
        int dc[4] = {0, 1, 0, -1};
        
        bool in_bounds(int r, int c); 

        // Add required elements to blocked_movement, then remove if the addition removes the existing path to the end
        void add_blocked_movement(int r1, int c1, int r2, int c2); 
        void remove_blocked_movement(int r1, int c1, int r2, int c2); 

        bool can_move_between(int r, int c, int n_r, int n_c); 

        bool has_path(int player); 
    
    public: 
        Quoridor(); 

        bool move_pawn(int player, int n_r, int n_c); 
        bool place_wall(int player, int r, int c, char orientation); 
        bool winner(int player); 
        void print_board(); 
}; 