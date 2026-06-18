#include "quoridor.h" 

int main() {
    Quoridor game; 
    int turn = 0; 

    while (1) {
        game.print_board(); 

        std::cout << "\nPlayer " << (turn == 0 ? 'A' : 'B') << "'s turn\n";
        std::cout << "Enter command:\n";
        std::cout << "  M row col        move pawn\n";
        std::cout << "  W row col H/V    place wall\n";
        std::cout << "> ";

        char cmd;
        std::cin >> cmd;
        cmd = std::toupper(cmd);

        bool ok = false;

        if (cmd == 'M') {
            int r, c;
            std::cin >> r >> c;
            ok = game.move_pawn(turn, r, c);
        } else if (cmd == 'W') {
            int r, c;
            char type;
            std::cin >> r >> c >> type;
            ok = game.place_wall(turn, r, c, type);
        }

        if (!ok) {
            std::cout << "Illegal move. Try again.\n";
            continue;
        }

        if (game.winner(turn)) {
            game.print_board();
            std::cout << "Player " << (turn == 0 ? 'A' : 'B') << " wins!\n";
            break;
        }

        turn = 1 - turn; 
    }

    return 0; 
}