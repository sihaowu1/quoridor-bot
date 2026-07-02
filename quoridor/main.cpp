// Interactive two-player CLI for the Quoridor engine.  Coordinates are
// ABSOLUTE board coordinates (row 0 at the top); the engine translates
// them into its canonical action space via action_for_move/action_for_wall.
//
// Build (no system compiler needed once `uv sync` has run):
//   .venv/lib/python*/site-packages/ziglang/zig c++ -std=c++17 -O2 \
//       quoridor/quoridor.cpp quoridor/main.cpp -o quoridor/quoridor

#include <cctype>
#include <iostream>
#include <limits>
#include <string>

#include "quoridor.h"

namespace {

void discard_line() {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

}  // namespace

int main(int argc, char** argv) {
    int board_size = argc > 1 ? std::atoi(argv[1]) : 9;
    int walls = argc > 2 ? std::atoi(argv[2]) : 10;
    quoridor::Engine game(board_size, walls);

    while (!game.done()) {
        char mover = game.to_play() == 1 ? 'X' : 'O';
        std::cout << '\n' << game.to_string() << "\n\n"
                  << "Player " << mover << "'s turn\n"
                  << "  M row col        move pawn (absolute coordinates)\n"
                  << "  W row col H/V    place wall at slot (row, col)\n"
                  << "  Q                quit\n"
                  << "> ";

        char cmd;
        if (!(std::cin >> cmd)) break;
        cmd = std::toupper(cmd);

        int action = -1;
        if (cmd == 'Q') {
            return 0;
        } else if (cmd == 'M') {
            int r, c;
            if (std::cin >> r >> c) action = game.action_for_move(r, c);
        } else if (cmd == 'W') {
            int r, c;
            char type;
            if (std::cin >> r >> c >> type) {
                action = game.action_for_wall(r, c, std::toupper(type));
            }
        }

        if (action < 0) {
            std::cout << "Illegal move. Try again.\n";
            discard_line();
            continue;
        }
        game.step(action);
    }

    if (!game.done()) return 0;  // EOF on stdin mid-game

    std::cout << '\n' << game.to_string() << '\n';
    if (game.winner() == 0) {
        std::cout << "Draw (move limit reached).\n";
    } else {
        std::cout << "Player " << (game.winner() == 1 ? 'X' : 'O')
                  << " wins!\n";
    }
    return 0;
}
