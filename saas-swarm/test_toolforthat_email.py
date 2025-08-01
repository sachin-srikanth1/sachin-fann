#!/usr/bin/env python3
"""
Test Email Configuration for toolforthat.io

This script tests common SMTP settings for custom domains.
"""

import smtplib
from email.message import EmailMessage

def test_smtp_config(server, port, username, password, use_tls=True, use_ssl=False):
    """Test SMTP configuration."""
    print(f"\n🔧 Testing {server}:{port} (TLS: {use_tls}, SSL: {use_ssl})")
    
    try:
        if use_ssl:
            server_obj = smtplib.SMTP_SSL(server, port)
        else:
            server_obj = smtplib.SMTP(server, port)
            if use_tls:
                server_obj.starttls()
        
        server_obj.login(username, password)
        
        # Create test email
        msg = EmailMessage()
        msg["From"] = username
        msg["To"] = username  # Send to yourself
        msg["Subject"] = "SaaS-Swarm Test Email"
        msg.set_content("This is a test email from SaaS-Swarm!")
        
        server_obj.send_message(msg)
        server_obj.quit()
        
        print("✅ SUCCESS!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    """Test common SMTP configurations for toolforthat.io."""
    email = "hello@toolforthat.io"
    password = input(f"Enter password for {email}: ").strip()
    
    print("📧 Testing toolforthat.io Email Configuration")
    print("=" * 60)
    
    # Common configurations to test
    configs = [
        # Google Workspace
        ("smtp.gmail.com", 587, True, False, "Google Workspace (Gmail)"),
        ("smtp.gmail.com", 465, False, True, "Google Workspace (Gmail SSL)"),
        
        # Microsoft 365
        ("smtp-mail.outlook.com", 587, True, False, "Microsoft 365"),
        ("smtp-mail.outlook.com", 465, False, True, "Microsoft 365 SSL"),
        
        # Custom hosting
        ("mail.toolforthat.io", 587, True, False, "Custom hosting (TLS)"),
        ("mail.toolforthat.io", 465, False, True, "Custom hosting (SSL)"),
        ("smtp.toolforthat.io", 587, True, False, "Custom SMTP (TLS)"),
        ("smtp.toolforthat.io", 465, False, True, "Custom SMTP (SSL)"),
        
        # Alternative ports
        ("mail.toolforthat.io", 25, False, False, "Custom hosting (Plain)"),
        ("smtp.toolforthat.io", 25, False, False, "Custom SMTP (Plain)"),
    ]
    
    working_configs = []
    
    for server, port, use_tls, use_ssl, description in configs:
        success = test_smtp_config(server, port, email, password, use_tls, use_ssl)
        if success:
            working_configs.append((server, port, use_tls, use_ssl, description))
    
    if working_configs:
        print(f"\n🎉 Found {len(working_configs)} working configuration(s)!")
        print("\nUse this in your .env file:")
        
        # Use the first working config
        server, port, use_tls, use_ssl, description = working_configs[0]
        
        print(f"""
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here

# Email Configuration ({description})
EMAIL_SMTP_SERVER={server}
EMAIL_SMTP_PORT={port}
EMAIL_USERNAME={email}
EMAIL_PASSWORD={password}
EMAIL_FROM_ADDRESS={email}
EMAIL_USE_TLS={str(use_tls).lower()}
        """)
        
        if len(working_configs) > 1:
            print("\nOther working configurations:")
            for server, port, use_tls, use_ssl, desc in working_configs[1:]:
                print(f"  - {desc}: {server}:{port}")
    
    else:
        print("\n❌ No working configurations found.")
        print("\nNext steps:")
        print("1. Check your email client settings for SMTP server info")
        print("2. Contact your email provider for SMTP settings")
        print("3. Try using a different email service (Gmail, Outlook)")
        print("4. Check if you need to enable SMTP access in your email settings")

if __name__ == "__main__":
    main() 