#include "rlagent.cuh"
#include <vector>

namespace mancalaCuda
{
/*
every cuda block has it's own simulation of the game it's playing through
each sim is just an array npits*2 + 2 large of ints plus a flag indicating who's turn it is
*/
	__device__ bool take_turn(board_state & bs, int action, bool & turnval)
	{
        //parameters for the turn
        int start_index = turnval*(nPits_player +1) + action;
        int pool_index = (turnval + 1) *(nPits_player + 1) - 1;
        //take the turn
        int beads = bs.pits[start_index];
        bs.pits[start_index] = 0;
        int index = start_index;
        for (int j =0; j < beads; j++)
        {
            index ++;
            if (index >= nPits_total)
            {
                index = 0;
            }
            bs.pits[index] ++;
        }
        if (index != pool_index)
        {
            turnval = !turnval;
        }
        //empty pot handling
        if (bs.pits[index] == 1 && index >= turnval*(nPits_player + 1) && index < pool_index)
        {
            int opp_index = (nPits_player * 2 - index) % nPits_total;
            bs.pits[pool_index] += bs.pits[opp_index] + bs.pits[index];
            bs.pits[opp_index] = 0;
            bs.pits[index] = 0;
        }

        for(int j = 0; j < nPits_player; j++)
        {
            if (bs.player1pits[j] > 0 || bs.player2pits[j] > 0)
            {
                return false;
            }
        }
        return true;
	}

    __device__ int chooseAction(board_state bs, bool player)
    {
        //for test just return first valid
        int playerInd = player ? nPits_player + 1 : 0;
        for(int i = 0; i < nPits_player; i++)
        {
            if(bs.pits[playerInd + i] > 0)
            {
                return i;
            }
        }
        return 0;
    }

    __global__ void playGame(int num_sims, int nturns, turn_record * results)
    {
		int run_index = blockIdx.x * blockDim.x + threadIdx.x;
		int run_stride = blockDim.x * gridDim.x;
		for (int i = run_index; i < num_sims; i += run_stride)
		{
            bool player = false;
            bool newgame = true;
            board_state board;
            for(int t = 0; t < nturns; t++)
            {
                if(newgame)
                {
                    for(int p = 0; p < nPits_player; p++)
                    {
                        board.player1pits[p] = nSeeds;
                        board.player2pits[p] = nSeeds;
                    }
                    board.player1pool = 0;
                    board.player2pool = 0;
                    newgame = false;
                }

                results[nturns*i + t].state = board;
                results[nturns*i + t].player = player;
                 //if a game finishes start a new one, we can finish the sim mid step
                int action = chooseAction(board, player);
                newgame = take_turn(board, action, player);
                results[nturns*i + t].action = action;
                if(newgame)
                {
                    for(int p = 0; p < nPits_player; p++)
                    {
                        board.player1pool += board.player1pits[p];
                        board.player2pool += board.player2pits[p];
                    }
                    results[nturns * i + t].reward = board.player1pool > board.player2pool ? 1 : 
                                                (board.player1pool < board.player2pool ? -1 : -2);
                }
                else
                {
                   results[nturns*i + t].reward = 0;
                }
            }
        }
    }



    void RLagent::parseBoardState(board_state& state, std::ostream & stream)
    {
        stream << "    |";
        for(int i = 0; i < nPits_player; i++)
        {
            stream << state.player1pits[i]  << "|"; 
        }
        stream << std::endl;
        stream << " |" << state.player1pool << "| ";
        for(int i = 0; i < nPits_player; i++)
        {
            stream << "  ";
        }
        stream << "|" << state.player2pool << "| ";
        stream << std::endl;
        stream << "    |";
        for(int i = 0; i < nPits_player; i++)
        {
            stream << state.player2pits[i]  << "|"; 
        }
        stream << std::endl;
    }

    RLagent::RLagent()
	{
		name = "rlagent";
	}
	
	std::string RLagent::GetName()
	{
		return name;
	}



    void RLagent::TrainStep()
    {
        int num_sims = 100000;
        int num_turns = 200;

        int num_records = num_sims*num_turns;
        int record_size = num_records * sizeof(turn_record);

        std::vector<turn_record> h_turnRecord(num_records);
        
       turn_record * d_turnRecord;
       cudaMalloc(&d_turnRecord, record_size);
        
       int threadsPerBlock = 32;
       int blocksPerGrid = (num_sims + threadsPerBlock - 1) / threadsPerBlock;

        playGame<<<blocksPerGrid, threadsPerBlock>>>(num_sims, num_turns, d_turnRecord);
        //playGame(num_sims, num_turns, h_turnRecord.data());


        cudaMemcpy(h_turnRecord.data(), d_turnRecord, record_size, cudaMemcpyDeviceToHost);

        cudaFree(d_turnRecord);

        //print the game log for checks
        for(int i = 0; i < num_turns; i++)
        {
            std::cout << "turn " << i << " action: " << h_turnRecord[i].action << " player: " <<
                h_turnRecord[i].player << " reward: " << h_turnRecord[i].reward << std::endl;

            parseBoardState(h_turnRecord[i].state, std::cout); 
            std::cout << std::endl;
        }
    }
}