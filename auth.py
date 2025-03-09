import os
import robin_stocks.robinhood as rh
import nacl.signing
import base64
import json
import datetime
import uuid
import requests
import configparser

def login_with_credentials():
    """
    Log into Robinhood using credentials stored in config.ini
    """
    config = configparser.ConfigParser()
    
    # Check if config.ini exists
    if not os.path.exists('config.ini'):
        username = input("Enter Robinhood username/email: ")
        password = input("Enter Robinhood password: ")
        
        # Create config file
        config['ROBINHOOD'] = {
            'username': username,
            'password': password
        }
        
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        print("Credentials saved to config.ini")
    else:
        config.read('config.ini')
        username = config['ROBINHOOD']['username']
        password = config['ROBINHOOD']['password']
    
    # Login to Robinhood
    login = rh.login(username, password)
    return login

def login_with_api_key():
    """
    Log into Robinhood using API key authentication
    Note: This is for API access which may require specific permissions
    """
    config = configparser.ConfigParser()
    
    # Check if config.ini exists with API keys
    if not os.path.exists('config.ini') or 'API' not in config:
        api_key = input("Enter your Robinhood API key: ")
        private_key_base64 = input("Enter your private key (base64 encoded): ")
        
        # Create or update config file
        if not config.has_section('API'):
            config.add_section('API')
            
        config['API'] = {
            'api_key': api_key,
            'private_key': private_key_base64
        }
        
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        print("API credentials saved to config.ini")
    else:
        config.read('config.ini')
        api_key = config['API']['api_key']
        private_key_base64 = config['API']['private_key']
    
    # Return API credentials for use in API calls
    return {
        'api_key': api_key,
        'private_key': private_key_base64
    }

def generate_key_pair():
    """
    Generate a new Ed25519 key pair for API authentication
    """
    private_key = nacl.signing.SigningKey.generate()
    public_key = private_key.verify_key
    
    # Convert keys to base64 strings
    private_key_base64 = base64.b64encode(private_key.encode()).decode()
    public_key_base64 = base64.b64encode(public_key.encode()).decode()
    
    print("Private Key (Base64):")
    print(private_key_base64)
    print("\nPublic Key (Base64):")
    print(public_key_base64)
    print("\nSAVE YOUR PRIVATE KEY SECURELY AND NEVER SHARE IT.")
    print("Use the public key when creating API credentials in Robinhood.")
    
    return private_key_base64, public_key_base64

def get_auth_headers(api_key, private_key_base64, method, path, body=""):
    """
    Generate authentication headers for Robinhood API requests
    """
    # Decode the private key from base64
    private_key_seed = base64.b64decode(private_key_base64)
    private_key = nacl.signing.SigningKey(private_key_seed)
    
    # Get current timestamp
    timestamp = int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())
    
    # Create the message to sign
    message_to_sign = f"{api_key}{timestamp}{path}{method}{body}"
    
    # Sign the message
    signed = private_key.sign(message_to_sign.encode("utf-8"))
    
    # Create and return the headers
    headers = {
        "x-api-key": api_key,
        "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
        "x-timestamp": str(timestamp),
        "Content-Type": "application/json"
    }
    
    return headers

if __name__ == "__main__":
    # Example of generating a new key pair
    private_key, public_key = generate_key_pair()
