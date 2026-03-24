import sys
import json
import traceback
import importlib.util
import tempfile
import os

def validate_environment(source_code: str, env_name: str) -> dict:
    result = {"valid": False, "error": None, "observation_space": None, "action_space": None}

    try:
        # Write source to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(source_code)
            tmp_path = f.name

        # Load module from file
        spec = importlib.util.spec_from_file_location("custom_env", tmp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find gymnasium.Env subclass
        import gymnasium as gym
        env_class = None
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, gym.Env) and obj != gym.Env:
                env_class = obj
                break

        if env_class is None:
            result["error"] = "No gymnasium.Env subclass found in source code"
            return result

        # Instantiate and test
        env = env_class()
        obs, info = env.reset()
        action = env.action_space.sample()
        env.step(action)
        env.close()

        result["valid"] = True
        result["observation_space"] = str(env.observation_space)
        result["action_space"] = str(env.action_space)
        os.unlink(tmp_path)

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result

if __name__ == "__main__":
    source_code = sys.stdin.read()
    env_name = sys.argv[1] if len(sys.argv) > 1 else "CustomEnv"
    result = validate_environment(source_code, env_name)
    print(json.dumps(result))
