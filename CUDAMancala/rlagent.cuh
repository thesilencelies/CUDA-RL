#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
//#include "Protobuf/TabularRLData.pb.h"

namespace mancalaCuda
{
	constexpr int nPits_player = 3;
	constexpr int nPits_total = (nPits_player + 1) * 2;
	constexpr int nStates_pit = nPits_total + 1;
	constexpr int nSeeds = 3;
	
	union board_state
	{
		int pits[nPits_total];
		struct
		{
		int player1pits[nPits_player];
		int player1pool;
		int player2pits[nPits_player];
		int player2pool;
		};
	};

    struct turn_record
    {
		board_state state;
        int action;
		// 1 for player1 win, -1 for player 2 win, -2 for draw, 0 otherwise
        int reward;
		bool player;
    };



	class RLagent
	{
		private:
		std::string name;
        int num_sims = 100000;
        int num_turns = 200;
        int num_records;
        int record_size;
        int num_states;
        int state_size;

		//cpu data
        std::vector<turn_record> h_turnRecord;
        std::vector<float> h_Qvals;

		//device data
		turn_record * d_turnRecord;
		float * d_Qvals;

		//helper functions
		void parseBoardState(board_state & state, std::ostream & stream);

		public:
		RLagent(int num_sims = 10000, int num_turns = 200);
		~RLagent();
		
		std::string GetName();
		
		void RunStep();
		void TrainStep();
		
		std::string PrintRun();

		void SaveQMat(std::string fileLoc);
		void LoadQMat(std::string fileLoc);
	};
}