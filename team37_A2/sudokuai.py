#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import copy
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        our_player = game_state.current_player
        
        all_moves = self.get_all_legal_moves(game_state)
        if not all_moves:
            return
        
        safe_moves, risky_moves = self.categorize_moves_by_safety(game_state, all_moves)
        ordered_moves = safe_moves + risky_moves
        
        if ordered_moves:
            self.propose_move(ordered_moves[0])
        
        for depth in range(1, 15):
            best_move = None
            best_value = float('-inf')
            alpha = float('-inf')
            beta = float('inf')
            
            search_moves = ordered_moves[:min(len(ordered_moves), 15 + depth * 3)]
            
            for move in search_moves:
                new_state = self.simulate_move(game_state, move)
                value = self.minimax(new_state, depth - 1, alpha, beta, False, our_player)
                
                if value > best_value:
                    best_value = value
                    best_move = move
                
                alpha = max(alpha, value)
            
            if best_move:
                self.propose_move(best_move)

    def get_candidates(self, board: SudokuBoard, square: tuple, taboo_moves: list) -> set:
        row, col = square
        N = board.N
        n = board.n
        m = board.m
        
        if board.get(square) != SudokuBoard.empty:
            return set()
        
        candidates = set(range(1, N + 1))
        
        for j in range(N):
            val = board.get((row, j))
            candidates.discard(val)
        
        for i in range(N):
            val = board.get((i, col))
            candidates.discard(val)
        
        block_row_start = (row // m) * m
        block_col_start = (col // n) * n
        for i in range(block_row_start, block_row_start + m):
            for j in range(block_col_start, block_col_start + n):
                val = board.get((i, j))
                candidates.discard(val)
        
        for taboo in taboo_moves:
            if taboo.square == square:
                candidates.discard(taboo.value)
        
        return candidates

    def is_move_safe(self, game_state: GameState, move: Move) -> bool:
        board = game_state.board
        N = board.N
        n = board.n
        m = board.m
        row, col = move.square
        value = move.value
        
        for j in range(N):
            if j == col:
                continue
            if board.get((row, j)) == SudokuBoard.empty:
                candidates = self.get_candidates(board, (row, j), game_state.taboo_moves)
                candidates.discard(value)
                if len(candidates) == 0:
                    return False
        
        for i in range(N):
            if i == row:
                continue
            if board.get((i, col)) == SudokuBoard.empty:
                candidates = self.get_candidates(board, (i, col), game_state.taboo_moves)
                candidates.discard(value)
                if len(candidates) == 0:
                    return False
        
        block_row_start = (row // m) * m
        block_col_start = (col // n) * n
        for i in range(block_row_start, block_row_start + m):
            for j in range(block_col_start, block_col_start + n):
                if i == row and j == col:
                    continue
                if board.get((i, j)) == SudokuBoard.empty:
                    candidates = self.get_candidates(board, (i, j), game_state.taboo_moves)
                    candidates.discard(value)
                    if len(candidates) == 0:
                        return False
        
        return True

    def count_remaining_candidates_after_move(self, game_state: GameState, move: Move) -> int:
        board = game_state.board
        N = board.N
        n = board.n
        m = board.m
        row, col = move.square
        value = move.value
        
        total_candidates = 0
        checked = set()
        
        for j in range(N):
            if j == col:
                continue
            sq = (row, j)
            if sq not in checked and board.get(sq) == SudokuBoard.empty:
                checked.add(sq)
                candidates = self.get_candidates(board, sq, game_state.taboo_moves)
                candidates.discard(value)
                total_candidates += len(candidates)
        
        for i in range(N):
            if i == row:
                continue
            sq = (i, col)
            if sq not in checked and board.get(sq) == SudokuBoard.empty:
                checked.add(sq)
                candidates = self.get_candidates(board, sq, game_state.taboo_moves)
                candidates.discard(value)
                total_candidates += len(candidates)
        
        block_row_start = (row // m) * m
        block_col_start = (col // n) * n
        for i in range(block_row_start, block_row_start + m):
            for j in range(block_col_start, block_col_start + n):
                if i == row and j == col:
                    continue
                sq = (i, j)
                if sq not in checked and board.get(sq) == SudokuBoard.empty:
                    checked.add(sq)
                    candidates = self.get_candidates(board, sq, game_state.taboo_moves)
                    candidates.discard(value)
                    total_candidates += len(candidates)
        
        return total_candidates

    def find_naked_singles(self, game_state: GameState, allowed_squares: list) -> list:
        board = game_state.board
        naked_singles = []
        
        for square in allowed_squares:
            if board.get(square) != SudokuBoard.empty:
                continue
            candidates = self.get_candidates(board, square, game_state.taboo_moves)
            if len(candidates) == 1:
                value = list(candidates)[0]
                move = Move(square, value)
                if self.is_move_safe(game_state, move):
                    naked_singles.append(move)
        
        return naked_singles

    def find_hidden_singles(self, game_state: GameState, allowed_squares: list) -> list:
        board = game_state.board
        N = board.N
        n = board.n
        m = board.m
        hidden_singles = []
        found = set()
        
        allowed_set = set(allowed_squares)
        
        for row in range(N):
            row_squares = [(row, j) for j in range(N) 
                           if board.get((row, j)) == SudokuBoard.empty and (row, j) in allowed_set]
            self._find_hidden_in_region(game_state, row_squares, hidden_singles, found)
        
        for col in range(N):
            col_squares = [(i, col) for i in range(N) 
                           if board.get((i, col)) == SudokuBoard.empty and (i, col) in allowed_set]
            self._find_hidden_in_region(game_state, col_squares, hidden_singles, found)
        
        for block_row in range(0, N, m):
            for block_col in range(0, N, n):
                block_squares = []
                for i in range(block_row, block_row + m):
                    for j in range(block_col, block_col + n):
                        if board.get((i, j)) == SudokuBoard.empty and (i, j) in allowed_set:
                            block_squares.append((i, j))
                self._find_hidden_in_region(game_state, block_squares, hidden_singles, found)
        
        return hidden_singles

    def _find_hidden_in_region(self, game_state: GameState, squares: list, 
                                hidden_singles: list, found: set) -> None:
        board = game_state.board
        N = board.N
        value_positions = {v: [] for v in range(1, N + 1)}
        
        for square in squares:
            candidates = self.get_candidates(board, square, game_state.taboo_moves)
            for val in candidates:
                value_positions[val].append(square)
        
        for val, positions in value_positions.items():
            if len(positions) == 1:
                square = positions[0]
                key = (square, val)
                if key not in found:
                    move = Move(square, val)
                    if self.is_move_safe(game_state, move):
                        found.add(key)
                        hidden_singles.append(move)

    def categorize_moves_by_safety(self, game_state: GameState, all_moves: list) -> tuple:
        allowed_squares = game_state.player_squares()
        if allowed_squares is None:
            N = game_state.board.N
            allowed_squares = [(i, j) for i in range(N) for j in range(N)]
        
        naked = self.find_naked_singles(game_state, allowed_squares)
        hidden = self.find_hidden_singles(game_state, allowed_squares)
        
        guaranteed_set = {(m.square, m.value) for m in naked + hidden}
        
        guaranteed = [m for m in all_moves if (m.square, m.value) in guaranteed_set]
        uncertain = [m for m in all_moves if (m.square, m.value) not in guaranteed_set]
        
        safe_uncertain = []
        risky_uncertain = []
        
        for move in uncertain:
            if self.is_move_safe(game_state, move):
                safe_uncertain.append(move)
            else:
                risky_uncertain.append(move)
        
        guaranteed = self.order_moves_by_safety(game_state, guaranteed)
        
        safe_uncertain = self.order_moves_by_safety(game_state, safe_uncertain)
        
        risky_uncertain = self.order_moves_by_score(game_state.board, risky_uncertain)
        
        return guaranteed + safe_uncertain, risky_uncertain

    def order_moves_by_safety(self, game_state: GameState, moves: list) -> list:
        def move_priority(move):
            score = self.calculate_move_score(game_state.board, move)
            flexibility = self.count_remaining_candidates_after_move(game_state, move)
            return (score * 100 + flexibility, -move.square[0], -move.square[1])
        
        return sorted(moves, key=move_priority, reverse=True)

    def order_moves_by_score(self, board: SudokuBoard, moves: list) -> list:
        def move_priority(move):
            score = self.calculate_move_score(board, move)
            return (score, -move.square[0], -move.square[1])
        
        return sorted(moves, key=move_priority, reverse=True)

    def get_all_legal_moves(self, game_state: GameState) -> list:
        board = game_state.board
        N = board.N
        
        allowed_squares = game_state.player_squares()
        if allowed_squares is None:
            allowed_squares = [(i, j) for i in range(N) for j in range(N) 
                               if board.get((i, j)) == SudokuBoard.empty]
        
        legal_moves = []
        
        for square in allowed_squares:
            if board.get(square) != SudokuBoard.empty:
                continue
            
            candidates = self.get_candidates(board, square, game_state.taboo_moves)
            for value in candidates:
                legal_moves.append(Move(square, value))
        
        return legal_moves

    def calculate_move_score(self, board: SudokuBoard, move: Move) -> int:
        N = board.N
        n = board.n
        m = board.m
        row, col = move.square
        
        regions_completed = 0
        
        row_empty = sum(1 for j in range(N) if board.get((row, j)) == SudokuBoard.empty)
        if row_empty == 1:
            regions_completed += 1
        
        col_empty = sum(1 for i in range(N) if board.get((i, col)) == SudokuBoard.empty)
        if col_empty == 1:
            regions_completed += 1
        
        block_row_start = (row // m) * m
        block_col_start = (col // n) * n
        block_empty = sum(1 for i in range(block_row_start, block_row_start + m)
                          for j in range(block_col_start, block_col_start + n)
                          if board.get((i, j)) == SudokuBoard.empty)
        if block_empty == 1:
            regions_completed += 1
        
        return [0, 1, 3, 7][regions_completed]

    def simulate_move(self, game_state: GameState, move: Move) -> GameState:
        new_board = copy.deepcopy(game_state.board)
        new_board.put(move.square, move.value)
        
        move_score = self.calculate_move_score(game_state.board, move)
        
        new_scores = list(game_state.scores)
        if game_state.current_player == 1:
            new_scores[0] += move_score
        else:
            new_scores[1] += move_score
        
        new_occupied1 = list(game_state.occupied_squares1) if game_state.occupied_squares1 else []
        new_occupied2 = list(game_state.occupied_squares2) if game_state.occupied_squares2 else []
        
        if game_state.current_player == 1:
            new_occupied1 = new_occupied1 + [move.square]
        else:
            new_occupied2 = new_occupied2 + [move.square]
        
        new_player = 2 if game_state.current_player == 1 else 1
        
        return GameState(
            initial_board=game_state.initial_board,
            board=new_board,
            taboo_moves=list(game_state.taboo_moves),
            moves=list(game_state.moves) + [move],
            scores=new_scores,
            current_player=new_player,
            allowed_squares1=game_state.allowed_squares1,
            allowed_squares2=game_state.allowed_squares2,
            occupied_squares1=new_occupied1,
            occupied_squares2=new_occupied2
        )

    def evaluate(self, game_state: GameState, our_player: int) -> float:
        score_diff = game_state.scores[0] - game_state.scores[1]
        if our_player == 2:
            score_diff = -score_diff
        
        all_moves = self.get_all_legal_moves(game_state)
        move_bonus = len(all_moves) * 0.01
        
        return score_diff + move_bonus

    def is_terminal(self, game_state: GameState) -> bool:
        board = game_state.board
        N = board.N
        for i in range(N):
            for j in range(N):
                if board.get((i, j)) == SudokuBoard.empty:
                    return False
        return True

    def minimax(self, game_state: GameState, depth: int, alpha: float, beta: float, 
                is_maximizing: bool, our_player: int) -> float:
        if depth == 0 or self.is_terminal(game_state):
            return self.evaluate(game_state, our_player)
        
        all_moves = self.get_all_legal_moves(game_state)
        if not all_moves:
            return self.evaluate(game_state, our_player)
        
        moves = self.order_moves_by_score(game_state.board, all_moves)
        
        moves = moves[:min(len(moves), 15)]
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in moves:
                new_state = self.simulate_move(game_state, move)
                eval_score = self.minimax(new_state, depth - 1, alpha, beta, False, our_player)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                new_state = self.simulate_move(game_state, move)
                eval_score = self.minimax(new_state, depth - 1, alpha, beta, True, our_player)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
