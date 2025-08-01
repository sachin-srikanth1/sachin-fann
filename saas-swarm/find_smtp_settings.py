#!/usr/bin/env python3
"""
Find SMTP Settings for Custom Domain

This script helps find the correct SMTP settings for custom domains
like toolforthat.io.
"""

import dns.resolver
import smtplib
from email.message import EmailMessage
import socket

def find_mx_records(domain):
    """Find MX records for a domain."""
    try:
        mx_records = dns.resolver.resolve(domain, 'MX')
        return [str(mx.exchange).rstrip('.') for mx in mx_records]
    except Exception as e:
        print(f"❌ Could not find MX records: {e}")
        return []

def find_smtp_server(domain):
    """Find SMTP server for a domain."""
    # Common SMTP server patterns
    smtp_servers = [
        f"smtp.{domain}",
        f"mail.{domain}",
        f"email.{domain}",
        f"mx.{domain}",
        f"smtp1.{domain}",
        f"smtp2.{domain}",
        f"mail1.{domain}",
        f"mail2.{domain}"
    ]
    
    # Add MX record servers
    mx_records = find_mx_records(domain)
    smtp_servers.extend(mx_records)
    
    working_servers = []
    
    for server in smtp_servers:
        try:
            # Try to connect to SMTP port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((server, 587))
            if result == 0:
                working_servers.append((server, 587))
            sock.close()
            
            # Try port 25
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((server, 25))
            if result == 0:
                working_servers.append((server, 25))
            sock.close()
            
            # Try port 465
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((server, 465))
            if result == 0:
                working_servers.append((server, 465))
            sock.close()
            
        except Exception:
            continue
    
    return working_servers

def test_smtp_connection(server, port, username, password):
    """Test SMTP connection and authentication."""
    print(f"\n🔧 Testing {server}:{port}")
    
    try:
        # Try with TLS
        if port == 587:
            server_obj = smtplib.SMTP(server, port)
            server_obj.starttls()
            server_obj.login(username, password)
            server_obj.quit()
            return True, "TLS"
        elif port == 465:
            server_obj = smtplib.SMTP_SSL(server, port)
            server_obj.login(username, password)
            server_obj.quit()
            return True, "SSL"
        else:
            server_obj = smtplib.SMTP(server, port)
            server_obj.login(username, password)
            server_obj.quit()
            return True, "Plain"
            
    except Exception as e:
        return False, str(e)

def main():
    """Find SMTP settings for toolforthat.io."""
    domain = "toolforthat.io"
    email = "hello@toolforthat.io"
    
    print(f"🔍 Finding SMTP settings for {domain}")
    print("=" * 50)
    
    # Find potential SMTP servers
    smtp_servers = find_smtp_server(domain)
    
    if not smtp_servers:
        print("❌ No SMTP servers found.")
        print("\nPossible solutions:")
        print("1. Check with your email provider for SMTP settings")
        print("2. Use a third-party email service (Gmail, Outlook, etc.)")
        print("3. Contact your domain administrator")
        return
    
    print(f"Found {len(smtp_servers)} potential SMTP servers:")
    for server, port in smtp_servers:
        print(f"  - {server}:{port}")
    
    # Get password
    password = input(f"\nEnter password for {email}: ").strip()
    
    # Test each server
    working_configs = []
    
    for server, port in smtp_servers:
        success, details = test_smtp_connection(server, port, email, password)
        
        if success:
            print(f"✅ {server}:{port} - {details}")
            working_configs.append((server, port, details))
        else:
            print(f"❌ {server}:{port} - {details}")
    
    if working_configs:
        print(f"\n🎉 Found {len(working_configs)} working configuration(s)!")
        print("\nUse this in your .env file:")
        
        # Use the first working config
        server, port, security = working_configs[0]
        
        if security == "TLS":
            print(f"""
EMAIL_SMTP_SERVER={server}
EMAIL_SMTP_PORT={port}
EMAIL_USERNAME={email}
EMAIL_PASSWORD={password}
EMAIL_FROM_ADDRESS={email}
EMAIL_USE_TLS=true
            """)
        elif security == "SSL":
            print(f"""
EMAIL_SMTP_SERVER={server}
EMAIL_SMTP_PORT={port}
EMAIL_USERNAME={email}
EMAIL_PASSWORD={password}
EMAIL_FROM_ADDRESS={email}
EMAIL_USE_TLS=false
            """)
        else:
            print(f"""
EMAIL_SMTP_SERVER={server}
EMAIL_SMTP_PORT={port}
EMAIL_USERNAME={email}
EMAIL_PASSWORD={password}
EMAIL_FROM_ADDRESS={email}
EMAIL_USE_TLS=false
            """)
    else:
        print("\n❌ No working configurations found.")
        print("\nTry these common settings:")
        print(f"""
# Option 1: Try with TLS
EMAIL_SMTP_SERVER=smtp.{domain}
EMAIL_SMTP_PORT=587
EMAIL_USERNAME={email}
EMAIL_PASSWORD={password}
EMAIL_FROM_ADDRESS={email}
EMAIL_USE_TLS=true

# Option 2: Try with SSL
EMAIL_SMTP_SERVER=smtp.{domain}
EMAIL_SMTP_PORT=465
EMAIL_USERNAME={email}
EMAIL_PASSWORD={password}
EMAIL_FROM_ADDRESS={email}
EMAIL_USE_TLS=false

# Option 3: Try without encryption
EMAIL_SMTP_SERVER=smtp.{domain}
EMAIL_SMTP_PORT=25
EMAIL_USERNAME={email}
EMAIL_PASSWORD={password}
EMAIL_FROM_ADDRESS={email}
EMAIL_USE_TLS=false
        """)

if __name__ == "__main__":
    main() 