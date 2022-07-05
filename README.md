# connect4-pybot
A Python agent for the game connect four based on the GYM kaggle env.

## Usage 

The script is a submission file for the [Connect X simulation comp](https://www.kaggle.com/c/connectx/overview) on Kaggle. 
It makes use of SOTA methods (AlphaZero) that you can check on the notebook. 


If you plan to make a submission on the competition, you will need to package the code with the model weights using the `make_sub.py` script. This will build an archive that you can then send to the competition.

Supposing you have computed the network weights in "weights.h5" and your agent definition is in "agent.py".
A usage example would be : `create_submission.py --weights-path weights.h5 --agent-path agent.py` 

