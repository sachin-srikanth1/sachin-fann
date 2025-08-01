#!/usr/bin/env python3
"""
Email Configuration Test Script

This script helps you test different email configurations
to find the right settings for your email provider.
"""

import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_email_config(provider_name, smtp_server, smtp_port, username, password, use_tls=True):
    """Test email configuration."""
    print(f"\n🔧 Testing {provider_name} configuration...")
    print(f"Server: {smtp_server}:{smtp_port}")
    print(f"Username: {username}")
    print(f"TLS: {use_tls}")
    
    try:
        # Create email message
        msg = EmailMessage()
        msg["From"] = username
        msg["To"] = username  # Send to yourself for testing
        msg["Subject"] = "SaaS-Swarm Email Test"
        msg.set_content("This is a test email from SaaS-Swarm!")
        
        # Connect to server
        if use_tls:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
        else:
            server = smtplib.SMTP(smtp_server, smtp_port)
        
        # Login
        server.login(username, password)
        
        # Send email
        server.send_message(msg)
        server.quit()
        
        print("✅ Email sent successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def main():
    """Test common email configurations."""
    print("📧 Email Configuration Test")
    print("=" * 50)
    
    # Get user's email
    email = input("Enter your email address: ").strip()
    password = input("Enter your email password: ").strip()
    
    # Common email providers
    providers = [
        {
            "name": "Gmail",
            "server": "smtp.gmail.com",
            "port": 587,
            "tls": True,
            "note": "⚠️  Gmail requires an App Password, not your regular password!"
        },
        {
            "name": "Outlook/Hotmail",
            "server": "smtp-mail.outlook.com",
            "port": 587,
            "tls": True,
            "note": "✅ Usually works with regular password"
        },
        {
            "name": "Yahoo",
            "server": "smtp.mail.yahoo.com",
            "port": 587,
            "tls": True,
            "note": "✅ Usually works with regular password"
        },
        {
            "name": "Gmail (App Password)",
            "server": "smtp.gmail.com",
            "port": 587,
            "tls": True,
            "note": "🔑 Use App Password from Google Account settings"
        }
    ]
    
    results = []
    
    for provider in providers:
        print(f"\n{provider['note']}")
        
        if provider['name'] == "Gmail (App Password)":
            use_app_password = input("Do you want to test with an App Password? (y/n): ").lower()
            if use_app_password != 'y':
                continue
        
        success = test_email_config(
            provider['name'],
            provider['server'],
            provider['port'],
            email,
            password,
            provider['tls']
        )
        
        results.append((provider['name'], success))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results")
    print("=" * 50)
    
    working_providers = []
    for name, success in results:
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"{name}: {status}")
        if success:
            working_providers.append(name)
    
    if working_providers:
        print(f"\n🎉 Working providers: {', '.join(working_providers)}")
        print("\nUse these settings in your .env file:")
        
        # Get the first working provider
        working_provider = working_providers[0]
        if "Gmail" in working_provider:
            print(f"""
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME={email}
EMAIL_PASSWORD=your-app-password-here
EMAIL_FROM_ADDRESS={email}
EMAIL_USE_TLS=true
            """)
        elif "Outlook" in working_provider:
            print(f"""
EMAIL_SMTP_SERVER=smtp-mail.outlook.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME={email}
EMAIL_PASSWORD={password}
EMAIL_FROM_ADDRESS={email}
EMAIL_USE_TLS=true
            """)
        elif "Yahoo" in working_provider:
            print(f"""
EMAIL_SMTP_SERVER=smtp.mail.yahoo.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME={email}
EMAIL_PASSWORD={password}
EMAIL_FROM_ADDRESS={email}
EMAIL_USE_TLS=true
            """)
    else:
        print("\n❌ No working configurations found.")
        print("\nTroubleshooting tips:")
        print("1. For Gmail: Use an App Password (not your regular password)")
        print("2. Check if your email provider allows SMTP access")
        print("3. Verify your password is correct")
        print("4. Try enabling 'Less secure app access' (if available)")

if __name__ == "__main__":
    main() 