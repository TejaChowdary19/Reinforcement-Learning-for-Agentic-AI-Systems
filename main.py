from envs.workflow_env import WorkflowEnv

if __name__ == "__main__":
    # Example task configuration
    task_config = {
        "name": "Summarize and Translate",
        "steps": ["Summarize", "Translate"],
        "tools": [
            {"name": "Tool A", "success_rate": 0.7},
            {"name": "Tool B", "success_rate": 0.5},
            {"name": "Tool C", "success_rate": 0.9}
        ]
    }

    env = WorkflowEnv(task_config)
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Random for now
        obs, reward, done, info = env.step(action)
        print(f"Step: {obs}, Reward: {reward}, Tool: {info['tool_used']}")
        env.render()
