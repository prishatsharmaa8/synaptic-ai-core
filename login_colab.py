import json
import os
import sys
from importlib import resources
from google_auth_oauthlib.flow import InstalledAppFlow
import colab_cli.auth as auth

print("Initializing seamless local browser sign-in for Google Colab CLI...")

config_resource = resources.files("colab_cli").joinpath("oauth_config.json")
client_config = json.loads(config_resource.read_text())

flow = InstalledAppFlow.from_client_config(client_config, auth.PUBLIC_SCOPES)
# Port 8085 matches localhost redirect
creds = flow.run_local_server(port=8085, prompt="consent", open_browser=True)

print("\nSuccessfully received OAuth credentials!")
token_json = creds.to_json()

# Save to Keyring
try:
    auth._save_token_to_keyring(token_json)
    print("Saved OAuth token to Windows keyring.")
except Exception as e:
    print(f"Keyring save notice: {e}")

# Also save to config file as fallback
os.makedirs(os.path.dirname(auth.LEGACY_TOKEN_PATH), exist_ok=True)
with open(auth.LEGACY_TOKEN_PATH, "w") as f:
    f.write(token_json)
print(f"Saved OAuth credentials to {auth.LEGACY_TOKEN_PATH}")

print("\nAuthentication complete! Google Colab CLI is now fully linked.")
