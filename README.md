conda create -n IQ python=3.9

conda activate IQ

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt

<!-- git clone https://github.com/Farama-Foundation/D4RL.git -->

cd D4RL

pip install -e .

cd ..

conda install -c conda-forge glew

conda install -c conda-forge mesalib

conda install -c menpo glfw3

export CPATH=$CONDA_PREFIX/include

pip install patchelf

```
# for 25000 expert transitions (25 trajectories)

python -u train_sqil.py env=ant agent=sac method=sqil \
expert.demos=25000 method.regularize=True \
agent.actor_lr=3e-05 agent.init_temp=0.001 seed=0 env.learn_steps=1e6

python -u train_iq.py env=ant agent=sac \
expert.demos=25000 method.loss=value method.regularize=True \
agent.actor_lr=3e-05 agent.init_temp=0.001 seed=0 env.learn_steps=1e6
```