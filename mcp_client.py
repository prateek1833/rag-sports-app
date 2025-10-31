# mcp_client.py (patch)
import requests

def _normalize_base(mcp_url: str) -> str:
    return mcp_url.rstrip("/")

def mcp_list_tools_sync(mcp_url: str):
    base = _normalize_base(mcp_url)
    try:
        response = requests.get(f"{base}/tools", timeout=5)
        response.raise_for_status()
        return response.json().get("tools", [])
    except requests.HTTPError as e:
        # include response body for debugging
        text = getattr(e.response, "text", None)
        raise RuntimeError(f"Failed to list tools: {e} - {text}")
    except Exception as e:
        raise RuntimeError(f"Failed to list tools: {e}")

def mcp_call_tool_sync(mcp_url: str, tool_name: str, parameters: dict):
    base = _normalize_base(mcp_url)
    try:
        payload = {"tool_name": tool_name, "parameters": parameters}
        response = requests.post(f"{base}/call_tool", json=payload, timeout=10)
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            # include server response text so you can see FastAPI detail
            raise RuntimeError(f"Failed to call tool '{tool_name}': {e} - {response.status_code} - {response.text}")
        return response.json().get("result")
    except Exception as e:
        raise RuntimeError(f"Failed to call tool '{tool_name}': {e}")
