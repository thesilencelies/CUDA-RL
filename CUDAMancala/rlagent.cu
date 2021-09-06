#include "rlagent.hpp"

namespace mancalaCuda
{
/*
every cuda block has it's own simulation of the game it's playing through
each sim is just an array npits*2 + 2 large of ints plus a flag indicating who's turn it is
*/




	__global__
	bool take_turn(int i, int* pits, int action, bool & turnval)
	{
        //parameters for the turn
        int game_index = i*nPits_total;
        int start_index = game_index + player*(nPits_player +1) + action;
        int pool_index = (player + 1) *(nPits_player + 1) - 1;
        //take the turn
        int beads = pits[start_index];
        pits[start_index] = 0;
        int index = start_index;
        for (int j =0; j < beads; j++)
        {
            index ++;
            if (index >= nPits_total)
            {
            index = game_index;
            }
            pits[index] ++;
        }
        if (index != pool_index)
        {
            turnval = !player;
        }
        //empty pot handling
        if (pits[index] == 1 && index >= player*(nPits_player + 1) && index < pool_index)
        {
            int opp_index = (nPits_player * 2 - index) % nPits_total + game_index;
            pits[pool_index] += pits[opp_index] + pits[index];
            pits[opp_index] = 0;
            pits[index] = 0;
        }
        //check if the game is over
        bool player1empty = true;
        bool player2empty = true;
        for(int j = 0; j < nPits_player; j++)
        {
            if (pits[game_index +j] > 0)
            {
            player1empty = false;
            break;
            }
        }

        for(int j = nPits_player + 1; j < nPits_player*2 + 1; j++)
        {
            if (pits[game_index +j] > 0)
            {
            player2empty = false;
            break;
            }
        }
        return player1empty || player2empty;
	}

    __global__ int chooseAction(int index, int* pits, bool player)
    {
        //for test just return first valid
        int playerInd = index + player ? nPits_player + 1 : 0;
        for(int i = 0; i < nPits_player; i++)
        {
            if(pits[playerInd + index] > 0)
            {
                return i;
            }
        }
        return 0;
    }

    __global__ void playGame(int nsims, int nturns, turn_record * results)
    {
		int run_index = blockIdx.x * blockDim.x + threadIdx.x;
		int run_stride = blockDim.x * gridDim.x;
		for (int i = run_index; i < num_sims; i += run_stride)
		{
            bool player = false;
            bool newgame = true;
            int game_index = i*nPits_total;
            for(int t = 0; t < nturns; t++)
            {
                if(newgame)
                {
                    for(int p = 0; p < nPits_total; p++)
                    {
                        pits[game_index + p] = (p + 1) % (nPits_player + 1) != 0 ? nSeeds : 0;
                    }
                    newgame = false;
                }

                  //if a game finishes start a new one, we can finish the sim mid step


            }
        }
    }




	RLagent::RLagent()
	{
		name = "rlagent";
	}
	
	std::string RLagent::GetName()
	{
		return name;
	}
}