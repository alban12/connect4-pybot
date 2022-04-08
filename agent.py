def my_agent(obs, config):
   
    ################################
    # Imports                      #
    ################################
    
    import numpy as np
    import random

    ################################
    # Constant and structures      #
    ################################
    
    N_STEPS=2
    
    #############     Helper variables for transposition table    #################
    n_random = 2 * 6 * 7 #2 * obs.col * obs.row
    randomValues=np.random.randint(100000,size=n_random)#Initialisation of random offline values
    randomValuesStorage=np.asarray(randomValues.reshape(6,14))
    transposition_table={}
    
    ################################
    # Helper functions             #
    ################################
    
    
    # Gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, piece, config):
        next_grid = grid.copy()
        for row in range(config.rows-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = piece
        return next_grid
    
    # Helper function for minimax: checks if agent or opponent has four in a row in the window
    def is_terminal_window(window, config):
        return window.count(1) == config.inarow or window.count(2) == config.inarow

    # Helper function for minimax: checks if game has ended
    def is_terminal_node(grid, config):
        # Check for draw 
        if list(grid[0, :]).count(0) == 0:
            return True
        # Check for win: horizontal, vertical, or diagonal
        # horizontal 
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if is_terminal_window(window, config):
                    return True
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if is_terminal_window(window, config):
                    return True
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        return False
    
    # Helper function for minimax: calculates value of heuristic for grid
    def get_heuristic(grid, mark, config):
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_threes_opp = count_windows(grid, 3, mark%2+1, config)
        num_fours_opp = count_windows(grid, 4, mark%2+1, config)
        score = num_threes - 1e2*num_threes_opp - 1e5*num_fours_opp + 1e6*num_fours
        return score
    
    # Helper function for get_heuristic: checks if window satisfies heuristic conditions
    def check_window(window, num_discs, piece, config):
        return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)

    # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
    def count_windows(grid, num_discs, piece, config):
        num_windows=0
        for row in range(config.rows):
            for col in range(config.columns):
                if col in range(config.columns-(config.inarow-1)):
                    horizontal_window = list(grid[row, col:col+config.inarow])
                    if check_window(horizontal_window, num_discs, piece, config):
                        num_windows += 1
                if row in range(config.rows-(config.inarow-1)):
                    vertical_window = list(grid[row:row+config.inarow, col])
                    if check_window(vertical_window, num_discs, piece, config):
                        num_windows += 1
                if (row in range(config.rows-(config.inarow-1))) & (col in range(config.columns-(config.inarow-1))):
                    positive_diagonal_window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                    if check_window(positive_diagonal_window, num_discs, piece, config):
                        num_windows += 1  
                if (row in range(config.inarow-1, config.rows)) & (col in range(config.columns-(config.inarow-1))):
                    negative_diagonal_window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                    if check_window(negative_diagonal_window, num_discs, piece, config):
                        num_windows += 1
        return num_windows

    
    # Helper function for NegaMax Fast (Transposition table) return the value,flag,depth for a given node if it exist 
    def transpositionTableLookup(node):
        zobrist_key=get_zb_hash(node)
        if zobrist_key in transposition_table:
            return transposition_table[zobrist_key]
        else:
            return [None,None,-1] #Return -1 for depth by convenience for now 
    
    # Helper function for NegaMax Fast (Transposition table) return the value,flag,depth for a given node if it exist 
   #def transpositionTableStore(node, ttEntry):
    #    zobrist_key=get_zb_hash(node)
     #   transposition_table[zobrist_key]=ttEntry
    
    
    #########################
    #        SOLVERS        #
    #########################
    
    # Uses alpha beta to calculate value of dropping piece in selected column
    def score_move_ab(grid, col, mark, config, nsteps):
        next_grid = drop_piece(grid, col, mark, config)#Child of max
        alpha=-np.Inf
        beta=np.Inf
        score = alphabeta(next_grid, nsteps-1, alpha, beta, False, mark, config)#We call alpha-beta to get what Min would do (maximizingPlayer=False)
        return score #The minmax value chosen by the opp 
    
    
    #########################
    # Agent makes selection #
    #########################

    # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
      
    # TODO : make use of weights from another files
      
    # Make use of the weight of the trained with Monte Carlo network by giving it the grid 
    # my_net = initiliaze the network with the weights 
    # move = my_net.predict(grid)
    # return move
    
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move_ab(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))    #Alpha_beta choice
    #scores = dict(zip(valid_moves, [score_move_ntt(grid, col, obs.mark, config, N_STEPS) for col in valid_moves])) #Negamax with TT choice
    #scores = dict(zip(valid_moves, [score_move_mtdf(grid, col, obs.mark, config, N_STEPS) for col in valid_moves])) # ID
    #scores = dict(zip(valid_moves, [score_move_mtdf(grid, col, obs.mark, config, N_STEPS) for col in valid_moves])) # Mtdf 
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)
