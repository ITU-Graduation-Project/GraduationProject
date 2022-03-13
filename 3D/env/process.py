"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn.functional as F
from collections import deque
from env.model import PPO
from env.SimulationEnv import SimulationEnv


def eval(opt, global_model, num_states, num_actions):
    torch.manual_seed(123)

    env = SimulationEnv()

    local_model = PPO(num_states, num_actions)
    if torch.cuda.is_available():
        local_model.cuda()
    local_model.eval()
    state = torch.from_numpy(env.reset())
    state = state.type(torch.FloatTensor)
    if torch.cuda.is_available():
        state = state.cuda()
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())

        logits, value = local_model(state)
        policy = F.softmax(logits)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)

        # Uncomment following lines if you want to save model whenever level is completed
        # if info["flag_get"]:
        #     print("Finished")
        #     torch.save(local_model.state_dict(),
        #                "{}/ppo_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, curr_step))

        env.render()
        actions.append(action)
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)
        state = state.type(torch.FloatTensor)
        if torch.cuda.is_available():
            state = state.cuda()
