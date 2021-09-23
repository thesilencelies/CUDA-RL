#include "Protobuf/TabularRLData.pb.h"
#include "rlagent.cuh"
#include <sstream>
#include <fstream>

namespace mancalaCuda
{
    //cuda functions

/*
every cuda block has it's own simulation of the game it's playing through
each sim is just an array npits*2 + 2 large of ints plus a flag indicating who's turn it is
*/

    //not very fast, but allows a space effient representation of state
    __device__ int getBoardIndex(board_state & bs)
    {
        int index = 0;
        int remainingSeeds = nSeeds_total;
        for(int i = nPits_total - 2; i >= 0; i--)
        {
            remainingSeeds -= bs.pits[i];
            int maxAvoidedIndex = 1;
            for(int j = 0; j < i; j++)
            {
                //apply the recursive sums - available states = (Sum)^j (n) => (n)(n+1)(n+2)etc/(i!)
                maxAvoidedIndex = (maxAvoidedIndex*(remainingSeeds + j))/(j+1);
            }
            index += maxAvoidedIndex;
        }
        return index;
    }

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

    __device__ int chooseAction(board_state & bs, bool player, const float* QMat)
    {
        int stateIndex = getBoardIndex(bs) * nPits_player;
        //explortation
        //start deterministic
        //choose based on Qmat
        float maxQ = -100000;
        int rval = 0;
        //for test just return first valid
        int playerInd = player ? nPits_player + 1 : 0;
        for(int i = 0; i < nPits_player; i++)
        {
            float qval = QMat[stateIndex + i];
            if(bs.pits[playerInd + i] > 0 &&  qval > maxQ)
            {
                rval = i;
                maxQ = qval;
            }
        }
        return rval;
    }

    __global__ void playGame(int num_sims, int nturns, turn_record * results, const float * QMat)
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
                int action = chooseAction(board, player, QMat);
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

    __global__ void updateQVals(int num_sims, int nturns, const turn_record * results,float * QMat)
    {
		int run_index = blockIdx.x * blockDim.x + threadIdx.x;
		int run_stride = blockDim.x * gridDim.x;
		for (int i = run_index; i < num_sims; i += run_stride)
		{
            ///TODO
        }
    }

    //class functions

    void RLAgent::parseBoardState(board_state& state, std::ostream & stream)
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

    RLAgent::RLAgent(int num_sims, int num_turns)
	{
		name = "rlagent";
        this->num_sims = num_sims;
        this->num_turns = num_turns;
        num_records = num_sims*num_turns;
        record_size = num_records * sizeof(turn_record);
        num_states = nPits_player;
        for(int j = 0; j < nPits_total -1; j++)
        {
            //apply the recursive sums - available states = (Sum)^j (n) => (n)(n+1)(n+2)etc/(i!)
            num_states = (num_states*(nSeeds_total + 1 + j))/(j+1);
        }
        state_size = num_states * sizeof(float);

        h_turnRecord.resize(num_records);
        h_Qvals.resize(num_states);
        cudaMalloc(&d_Qvals, state_size);
        cudaMalloc(&d_turnRecord, record_size);
        cudaMemcpy(d_Qvals, h_Qvals.data(), state_size, cudaMemcpyHostToDevice);
	}

    RLAgent::~RLAgent()
    {
        cudaFree(d_turnRecord);
        cudaFree(d_Qvals);
    }
	
	std::string RLAgent::GetName()
	{
		return name;
	}



    void RLAgent::RunStep()
    {
       int threadsPerBlock = 32;
       int blocksPerGrid = (num_sims + threadsPerBlock - 1) / threadsPerBlock;

        playGame<<<blocksPerGrid, threadsPerBlock>>>(num_sims, num_turns, d_turnRecord, d_Qvals );
    }

    void RLAgent::TrainStep()
    {
        
       int threadsPerBlock = 32;
       int blocksPerGrid = (num_sims + threadsPerBlock - 1) / threadsPerBlock;

        updateQVals<<<blocksPerGrid, threadsPerBlock>>>(num_sims, num_turns, d_turnRecord, d_Qvals );

    }

    std::string RLAgent::PrintRun()
    {
        cudaMemcpy(h_turnRecord.data(), d_turnRecord, record_size, cudaMemcpyDeviceToHost);
        std::stringstream outStream;
        for(int i = 0; i < num_turns; i++)
        {
            outStream << "turn " << i << " action: " << h_turnRecord[i].action << " player: " <<
                h_turnRecord[i].player << " reward: " << h_turnRecord[i].reward << std::endl;

            parseBoardState(h_turnRecord[i].state, outStream); 
            outStream << std::endl;
        }
        return outStream.str();
    }

    void RLAgent::SaveQMat(std::string fileLoc)
    {
        cudaMemcpy(h_Qvals.data(), d_Qvals, state_size, cudaMemcpyDeviceToHost);
        mancala::QAgent outputProt;
        outputProt.set_npits(nPits_player);
        outputProt.set_nseeds(nSeeds);
        outputProt.mutable_q()->Add(h_Qvals.begin(), h_Qvals.end());
        std::cout  << num_states << " "  << h_Qvals.size() << " "<< outputProt.mutable_q()->size();
        std::ofstream file(fileLoc);
        outputProt.SerializeToOstream(&file);
        file.close();
    }

    void RLAgent::LoadQMat(std::string fileLoc)
    {
        std::ifstream file(fileLoc);
        mancala::QAgent inputProt;
        inputProt.ParseFromIstream(&file);
        file.close();
        if(inputProt.q().size() == num_states)
        {
            h_Qvals.assign(inputProt.q().begin(), inputProt.q().end());
            cudaMemcpy(d_Qvals, h_Qvals.data(), state_size, cudaMemcpyHostToDevice);
        }
    }

    template<unsigned int blockSize>
    __device__ void warpReduceMax(volatile int * sdata, int tid)
    {
        if (blockSize >= 64) sdata[tid] = sdata[tid] > sdata[tid + 32] ? sdata[tid] : sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] = sdata[tid] > sdata[tid + 16] ? sdata[tid] : sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] = sdata[tid] > sdata[tid + 8] ? sdata[tid] : sdata[tid + 8];
        if (blockSize >= 8) sdata[tid] = sdata[tid] > sdata[tid + 4] ? sdata[tid] : sdata[tid + 4];
        if (blockSize >= 4) sdata[tid] = sdata[tid] > sdata[tid + 2] ? sdata[tid] : sdata[tid + 2];
        if (blockSize >= 2) sdata[tid] = sdata[tid] > sdata[tid + 1] ? sdata[tid] : sdata[tid + 1];
    }

    template<unsigned int blockSize>
    __global__ void reduceMax(int datalen, int *g_idata, int *g_odata) {
        extern __shared__ int sdata[];
        // each thread loads one element from global to shared mem
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
        unsigned int gridSize = blockSize*2*gridDim.x;
        sdata[tid] = INT_MIN;

        //serial reduction first until we come down to the grid size
        while(i < datalen)
        {
            int a = g_idata[i];
            sdata[tid] = sdata[tid] >  a ? sdata[tid] : a; 
            i += gridSize;
        }
        __syncthreads();
        // do reduction in shared mem
        if(blockSize >= 512) { 
            if (tid < 256) { 
            sdata[tid] = sdata[tid] > sdata[tid + 256] ? sdata[tid] : sdata[tid + 256];}
            __syncthreads();
        }
        if(blockSize >= 256) { 
            if (tid < 128) { 
            sdata[tid] = sdata[tid] > sdata[tid + 128] ? sdata[tid] : sdata[tid + 128];}
            __syncthreads();
        }
        if(blockSize >= 128) { 
            if (tid < 64) { 
            sdata[tid] = sdata[tid] > sdata[tid + 64] ? sdata[tid] : sdata[tid + 64];}
            __syncthreads();
        }

        if(tid < 32)
        {
            warpReduceMax<blockSize>(sdata, tid);
        }

        // write result for this block to global mem
        if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    }


    //for practicing reduce
    int RLAgent::GetMax(std::vector<int> values)
    {
        int dataSize = values.size()*sizeof(int);

        int* d_values;
        int* d_output;
        cudaMalloc(&d_values, dataSize);
        cudaMalloc(&d_output, dataSize);
        cudaMemcpy(d_values, values.data(), dataSize, cudaMemcpyHostToDevice);
        
        constexpr int blockSize = 128;
        constexpr int itemsPerThread = 128;

        constexpr int itemsPerBlock = blockSize*itemsPerThread;
        for(int i = values.size(); i > 1; i = i/ blockSize + 1)
        {
            int nBlock = (i + itemsPerBlock - 1)/(itemsPerBlock);
             reduceMax<blockSize><<<nBlock, blockSize, blockSize*sizeof(int)>>>(i, d_values, d_output);
            i = i/ blockSize + 1;
            if(i > 1)
            {
                nBlock = (i + itemsPerBlock - 1)/(itemsPerBlock);
                reduceMax<blockSize><<<nBlock, blockSize, blockSize*sizeof(int)>>>(i, d_output, d_values);
            }
            else
            {
                cudaDeviceSynchronize();
                cudaMemcpy(d_values, d_output, sizeof(int), cudaMemcpyDeviceToDevice);
                cudaDeviceSynchronize();
            }
        }
        int rval = -1;
        cudaMemcpy( &rval, d_values,sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaFree(d_values);
        cudaFree(d_output);
        return rval;
    }
}