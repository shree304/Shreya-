# generate dashboard.html for this code # app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import random
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# .html

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///energy_assistant.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    devices = db.relationship('Device', backref='owner', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Device(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    status = db.Column(db.Boolean, default=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    readings = db.relationship('EnergyReading', backref='device', lazy=True, cascade="all, delete-orphan")

class EnergyReading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    power_usage = db.Column(db.Float, nullable=False)  # in kWh
    device_id = db.Column(db.Integer, db.ForeignKey('device.id'), nullable=False)

class SolarProduction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    production = db.Column(db.Float, nullable=False)  # in kWh
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class BatteryStorage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    level = db.Column(db.Float, nullable=False)  # percentage
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Recommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    message = db.Column(db.Text, nullable=False)
    impact = db.Column(db.Float)  # estimated kWh savings
    implemented = db.Column(db.Boolean, default=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    message = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(20), nullable=False)  # 'low', 'medium', 'high'
    read = db.Column(db.Boolean, default=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class AutomationRule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    condition_type = db.Column(db.String(50), nullable=False)  # 'time', 'price', 'solar', 'battery'
    condition_value = db.Column(db.String(100), nullable=False)
    action_type = db.Column(db.String(50), nullable=False)  # 'turn_on', 'turn_off', 'adjust'
    action_value = db.Column(db.String(100))
    device_id = db.Column(db.Integer, db.ForeignKey('device.id'))
    enabled = db.Column(db.Boolean, default=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Helper functions for energy analysis and recommendations
def get_current_electricity_rate():
    # Simulate peak/off-peak pricing
    current_hour = datetime.now().hour
    if 17 <= current_hour <= 21:  # Peak hours 5pm-9pm
        return 0.30  # $0.30 per kWh during peak
    elif 7 <= current_hour < 17:  # Mid-peak 7am-5pm
        return 0.20  # $0.20 per kWh
    else:  # Off-peak overnight
        return 0.10  # $0.10 per kWh

def analyze_device_patterns(user_id):
    # Get devices and their usage patterns
    devices = Device.query.filter_by(user_id=user_id).all()
    patterns = {}
    
    for device in devices:
        # Get readings from the past week
        week_ago = datetime.utcnow() - timedelta(days=7)
        readings = EnergyReading.query.filter(
            EnergyReading.device_id == device.id,
            EnergyReading.timestamp >= week_ago
        ).all()
        
        # Calculate hourly averages
        hourly_usage = [0] * 24
        hourly_counts = [0] * 24
        
        for reading in readings:
            hour = reading.timestamp.hour
            hourly_usage[hour] += reading.power_usage
            hourly_counts[hour] += 1
        
        # Average out the usage per hour
        for hour in range(24):
            if hourly_counts[hour] > 0:
                hourly_usage[hour] /= hourly_counts[hour]
        
        patterns[device.name] = {
            'hourly_usage': hourly_usage,
            'total_daily_avg': sum(hourly_usage),
        }
    
    return patterns

def generate_ai_recommendations(user_id):
    """Generate recommendations using Gemini AI model if available, otherwise use rule-based recommendations"""
    if GEMINI_API_KEY:
        try:
            # Collect data to send to Gemini
            patterns = analyze_device_patterns(user_id)
            current_rate = get_current_electricity_rate()
            
            # Get recent solar production
            today = datetime.utcnow().date()
            yesterday = today - timedelta(days=1)
            yesterday_start = datetime.combine(yesterday, datetime.min.time())
            
            solar_data = SolarProduction.query.filter(
                SolarProduction.user_id == user_id,
                SolarProduction.timestamp >= yesterday_start
            ).all()
            
            solar_production = [{"timestamp": reading.timestamp.strftime("%Y-%m-%d %H:%M"), 
                               "production": reading.production} for reading in solar_data]
            
            # Get battery levels
            battery_data = BatteryStorage.query.filter(
                BatteryStorage.user_id == user_id,
                BatteryStorage.timestamp >= yesterday_start
            ).order_by(BatteryStorage.timestamp.desc()).all()
            
            battery_levels = [{"timestamp": reading.timestamp.strftime("%Y-%m-%d %H:%M"), 
                             "level": reading.level} for reading in battery_data]
            
            # Get device status
            devices = Device.query.filter_by(user_id=user_id).all()
            device_info = [{"name": device.name, "type": device.type, "status": device.status} 
                         for device in devices]
            
            # Prepare prompt for Gemini
            prompt = f"""
            As an AI-powered energy assistant, analyze the following home energy data and provide 3 specific, 
            actionable recommendations to optimize energy usage. Consider energy efficiency, cost savings, 
            and comfort.

            Current electricity rate: ${current_rate}/kWh
            
            Device usage patterns: {json.dumps(patterns, indent=2)}
            
            Recent solar production: {json.dumps(solar_production[-10:], indent=2)}
            
            Recent battery levels: {json.dumps(battery_levels[-10:], indent=2)}
            
            Device status: {json.dumps(device_info, indent=2)}
            
            For each recommendation:
            1. Provide specific device or behavior changes
            2. Estimate energy savings in kWh
            3. Explain the rationale
            
            Format each recommendation as: 
            Recommendation: [clear action statement]
            Impact: [estimated kWh savings]
            Rationale: [brief explanation]
            
            Only include the three recommendations, no introductions or conclusions.
            """
            
            # Generate recommendations with Gemini
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            
            # Parse the response
            if response and response.text:
                recommendations = []
                
                # Basic parsing of Gemini response - adjust as needed based on actual output format
                sections = response.text.split("Recommendation:")
                
                for section in sections[1:]:  # Skip first element which is empty
                    lines = section.strip().split("\n")
                    if len(lines) >= 3:
                        recommendation_text = lines[0].strip()
                        
                        # Extract impact (kWh)
                        impact_line = [line for line in lines if line.strip().startswith("Impact:")]
                        impact = 1.0  # Default value
                        if impact_line:
                            impact_text = impact_line[0].replace("Impact:", "").strip()
                            # Extract numeric value from text like "2.5 kWh"
                            try:
                                impact = float(''.join(c for c in impact_text if c.isdigit() or c == '.'))
                            except:
                                impact = 1.0
                        
                        recommendations.append({
                            'message': "Recommendation: " + recommendation_text,
                            'impact': impact
                        })
                
                # If parsing failed, provide fallback recommendations
                if not recommendations:
                    return generate_recommendations(user_id)
                
                # Save recommendations to database
                for rec in recommendations:
                    new_recommendation = Recommendation(
                        message=rec['message'],
                        impact=rec['impact'],
                        user_id=user_id
                    )
                    db.session.add(new_recommendation)
                
                try:
                    db.session.commit()
                except:
                    db.session.rollback()
                
                return recommendations
            
        except Exception as e:
            print(f"Error generating AI recommendations: {e}")
            # Fall back to rule-based recommendations
            return generate_recommendations(user_id)
    
    # If no Gemini API key or error occurred, use rule-based approach
    return generate_recommendations(user_id)

def generate_recommendations(user_id):
    # Get device usage patterns
    patterns = analyze_device_patterns(user_id)
    
    # Check current energy rates
    current_rate = get_current_electricity_rate()
    
    # Get solar production data
    today = datetime.utcnow().date()
    solar_data = SolarProduction.query.filter(
        SolarProduction.user_id == user_id,
        SolarProduction.timestamp >= today
    ).all()
    
    # Get battery level
    battery = BatteryStorage.query.filter_by(user_id=user_id).order_by(BatteryStorage.timestamp.desc()).first()
    
    recommendations = []
    
    # Example logic for recommendations
    high_usage_devices = [(name, data) for name, data in patterns.items() if data['total_daily_avg'] > 5]
    
    if high_usage_devices:
        for device_name, data in high_usage_devices:
            # Check if device is used during peak hours
            peak_usage = sum(data['hourly_usage'][17:22])  # 5pm to 9pm
            if peak_usage > 0:
                recommendations.append({
                    'message': f"Consider shifting {device_name} usage outside peak hours (5pm-9pm) to save on electricity costs.",
                    'impact': peak_usage * 0.10  # Savings from shifting peak to off-peak
                })
    
    # Solar optimization recommendations
    if solar_data:
        solar_production = sum(reading.production for reading in solar_data)
        if solar_production > 0:
            recommendations.append({
                'message': f"You've generated {solar_production:.2f} kWh of solar energy today. Consider running major appliances during daylight hours.",
                'impact': solar_production * 0.20  # Approximate savings
            })
    
    # Battery recommendations
    if battery and battery.level < 30:
        recommendations.append({
            'message': "Battery storage is low. Consider charging during off-peak hours tonight.",
            'impact': 2.0  # Approximate savings
        })
    
    # Save recommendations to database
    for rec in recommendations:
        new_recommendation = Recommendation(
            message=rec['message'],
            impact=rec['impact'],
            user_id=user_id
        )
        db.session.add(new_recommendation)
    
    try:
        db.session.commit()
    except:
        db.session.rollback()
    
    return recommendations

def detect_anomalies(user_id):
    # Get devices
    devices = Device.query.filter_by(user_id=user_id).all()
    alerts = []
    
    for device in devices:
        # Get recent readings
        recent_readings = EnergyReading.query.filter_by(device_id=device.id).order_by(EnergyReading.timestamp.desc()).limit(10).all()
        
        if recent_readings and len(recent_readings) > 3:
            # Calculate average and standard deviation
            values = [r.power_usage for r in recent_readings]
            avg = sum(values) / len(values)
            std_dev = (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5
            
            # Check latest reading
            latest = recent_readings[0]
            if latest.power_usage > avg + 2 * std_dev:
                alert_message = f"Unusual power spike detected for {device.name}: {latest.power_usage:.2f} kWh (normally around {avg:.2f} kWh)"
                
                # Create alert
                new_alert = Alert(
                    message=alert_message,
                    severity='high' if latest.power_usage > avg + 3 * std_dev else 'medium',
                    user_id=user_id
                )
                db.session.add(new_alert)
                alerts.append(new_alert)
    
    try:
        db.session.commit()
    except:
        db.session.rollback()
    
    return alerts

# Generate sample data for new users
def generate_sample_data(user_id):
    # Create sample devices
    device_types = [
        {"name": "Smart Thermostat", "type": "climate"},
        {"name": "Refrigerator", "type": "appliance"},
        {"name": "Washing Machine", "type": "appliance"},
        {"name": "Living Room Lights", "type": "lighting"},
        {"name": "Home Office", "type": "office"},
        {"name": "EV Charger", "type": "transport"},
        {"name": "Solar Panels", "type": "generation"},
        {"name": "Battery Storage", "type": "storage"}
    ]
    
    created_devices = []
    for device_info in device_types:
        device = Device(name=device_info["name"], type=device_info["type"], user_id=user_id)
        db.session.add(device)
        created_devices.append(device)
    
    db.session.commit()
    
    # Generate sample readings for the past week
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    
    # Device energy patterns (in kWh)
    patterns = {
        "Smart Thermostat": {"base": 0.5, "peak": 1.5, "variation": 0.3},
        "Refrigerator": {"base": 0.1, "peak": 0.3, "variation": 0.05},
        "Washing Machine": {"base": 0, "peak": 2.0, "variation": 0.5},
        "Living Room Lights": {"base": 0, "peak": 0.5, "variation": 0.1},
        "Home Office": {"base": 0, "peak": 1.0, "variation": 0.2},
        "EV Charger": {"base": 0, "peak": 7.0, "variation": 1.0},
    }
    
    current_time = start_date
    while current_time <= end_date:
        hour = current_time.hour
        day_of_week = current_time.weekday()  # 0-6 (Monday to Sunday)
        
        # Simulate different usage patterns based on time of day and day of week
        for device in created_devices:
            if device.name in patterns:
                pattern = patterns[device.name]
                
                # Skip non-active periods for some devices
                if device.name == "Washing Machine" and not (10 <= hour <= 14 or 18 <= hour <= 20):
                    continue
                if device.name == "Living Room Lights" and 8 <= hour <= 16:
                    continue
                if device.name == "Home Office" and not (9 <= hour <= 17) and day_of_week < 5:
                    continue
                if device.name == "EV Charger" and not (22 <= hour or hour <= 4):
                    continue
                
                # Calculate usage with time-of-day factors
                base_usage = pattern["base"]
                peak_factor = 0
                
                # Time-based patterns
                if device.name == "Smart Thermostat":
                    if 6 <= hour <= 9 or 17 <= hour <= 22:  # Morning and evening peaks
                        peak_factor = 0.8
                    elif 10 <= hour <= 16:  # Midday
                        peak_factor = 0.5
                    else:  # Night
                        peak_factor = 0.3
                elif device.name == "Refrigerator":
                    if 12 <= hour <= 14 or 18 <= hour <= 21:  # Meal times (door opening)
                        peak_factor = 0.9
                    else:
                        peak_factor = 0.4
                else:
                    peak_factor = 0.7  # Default for other devices
                
                # Calculate power usage with some randomness
                power_usage = base_usage + (pattern["peak"] - base_usage) * peak_factor
                power_usage += random.uniform(-pattern["variation"], pattern["variation"])
                power_usage = max(0, power_usage)  # Ensure non-negative
                
                # Create the reading
                reading = EnergyReading(
                    timestamp=current_time,
                    power_usage=power_usage,
                    device_id=device.id
                )
                db.session.add(reading)
        
        # Generate solar production (daytime only)
        if 7 <= hour <= 19:  # Daylight hours
            # More production in middle of day
            production_factor = 1 - abs(13 - hour) / 6  # Peak at 1pm
            base_production = 2.0 * production_factor
            
            # Add weather variation and randomness
            weather_factor = random.uniform(0.7, 1.0)  # Simulate cloud cover
            production = base_production * weather_factor
            production = max(0, production + random.uniform(-0.3, 0.3))
            
            solar = SolarProduction(
                timestamp=current_time,
                production=production,
                user_id=user_id
            )
            db.session.add(solar)
        
        # Generate battery storage readings
        battery_level = 50.0  # Initialize battery level at 50%
        current_time = start_date
        while current_time <= end_date:
           hour = current_time.hour
        
        # Update battery level based on time of day
        if 9 <= hour <= 16:  # Solar charging hours
            battery_level = min(95, battery_level + random.uniform(3, 7))
        elif 17 <= hour <= 23:  # Evening discharge
            battery_level = max(30, battery_level - random.uniform(4, 8))
        else:  # Slow overnight discharge
            battery_level = max(20, battery_level - random.uniform(1, 3))
        
        # Create battery reading
        battery = BatteryStorage(
            timestamp=current_time,
            level=battery_level,
            user_id=user_id
        )
        db.session.add(battery)
        
        current_time += timedelta(hours=1)
         
            
        
        # Battery charges during solar production and discharges evening/night
        # if hour == 0:  # Start of day
        #     battery_level = 50.0  # Start at 50%
        # elif 9 <= hour <= 16:  # Solar charging hours
        #     battery_level = min(95, battery_level + random.uniform(3, 7))
        # elif 17 <= hour <= 23:  # Evening discharge
        #     battery_level = max(30, battery_level - random.uniform(4, 8))
        # else:
        #     battery_level = battery_level - random.uniform(1, 3)  # Slow overnight discharge
        
        # battery = BatteryStorage(
        #     timestamp=current_time,
        #     level=battery_level,
        #     user_id=user_id
        # )
        # db.session.add(battery)
        
        # current_time += timedelta(hours=1)
        
        
        
    
    # Create some initial recommendations
    recommendations = [
        {"message": "Schedule your washing machine to run during solar production hours (10am-2pm) to maximize self-consumption.", "impact": 1.8},
        {"message": "Your EV charging is happening during peak hours. Consider shifting to overnight charging to save approximately $25/month.", "impact": 8.5},
        {"message": "We detected your refrigerator may be using more energy than typical models. Consider a maintenance check.", "impact": 2.4}
    ]
    
    for rec in recommendations:
        recommendation = Recommendation(
            message=rec["message"],
            impact=rec["impact"],
            user_id=user_id
        )
        db.session.add(recommendation)
    
    # Create initial alerts
    alerts = [
        {"message": "Unusual power spike detected from your Home Office (3.2 kWh). This is 210% higher than your typical usage.", "severity": "medium"},
        {"message": "Your battery storage is consistently below optimal levels during peak pricing hours. Consider adjusting your charging schedule.", "severity": "low"}
    ]
    
    for alert_info in alerts:
        alert = Alert(
            message=alert_info["message"],
            severity=alert_info["severity"],
            user_id=user_id
        )
        db.session.add(alert)
    
    # Create sample automation rules
    automation_rules = [
        {
            "name": "Night Thermostat Setback",
            "description": "Lower thermostat at night to save energy",
            "condition_type": "time",
            "condition_value": "22:00",
            "action_type": "adjust",
            "action_value": "68F",
            "device_name": "Smart Thermostat"
        },
        {
            "name": "Solar EV Charging",
            "description": "Charge EV when solar production is high",
            "condition_type": "solar",
            "condition_value": "1.5",
            "action_type": "turn_on",
            "action_value": "",
            "device_name": "EV Charger"
        }
    ]
    
    for rule_info in automation_rules:
        device = Device.query.filter_by(user_id=user_id, name=rule_info["device_name"]).first()
        if device:
            rule = AutomationRule(
                name=rule_info["name"],
                description=rule_info["description"],
                condition_type=rule_info["condition_type"],
                condition_value=rule_info["condition_value"],
                action_type=rule_info["action_type"],
                action_value=rule_info["action_value"],
                device_id=device.id,
                user_id=user_id
            )
            db.session.add(rule)
    
    db.session.commit()

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        # Generate sample data for the new user
        generate_sample_data(user.id)
        
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Generate new recommendations and check for anomalies
    if GEMINI_API_KEY:
        generate_ai_recommendations(current_user.id)
    else:
        generate_recommendations(current_user.id)
    detect_anomalies(current_user.id)
    
    # Get relevant data for dashboard
    energy_summary = get_energy_summary(current_user.id)
    recent_alerts = Alert.query.filter_by(user_id=current_user.id, read=False).order_by(Alert.timestamp.desc()).limit(5).all()
    recommendations = Recommendation.query.filter_by(user_id=current_user.id, implemented=False).order_by(Recommendation.timestamp.desc()).limit(5).all()
    
    # Get devices for the device status panel
    devices = Device.query.filter_by(user_id=current_user.id).all()
    
    # Get data for charts
    charts_data = generate_charts_data(current_user.id)
    
    return render_template(
        'dashboard.html',
        energy_summary=energy_summary,
        recent_alerts=recent_alerts,
        recommendations=recommendations,
        devices=devices,
        charts_data=charts_data
    )

def get_energy_summary(user_id):
    # Get today's date range
    today = datetime.utcnow().date()
    tomorrow = today + timedelta(days=1)
    today_start = datetime.combine(today, datetime.min.time())
    today_end = datetime.combine(tomorrow, datetime.min.time())
    
    # Get yesterday for comparison
    yesterday = today - timedelta(days=1)
    yesterday_start = datetime.combine(yesterday, datetime.min.time())
    yesterday_end = today_start
    
    # Calculate total usage today
    today_readings = EnergyReading.query.join(Device).filter(
        Device.user_id == user_id,
        EnergyReading.timestamp >= today_start,
        EnergyReading.timestamp < today_end
    ).all()
    
    today_usage = sum(reading.power_usage for reading in today_readings)
    
    # Calculate total usage yesterday
    yesterday_readings = EnergyReading.query.join(Device).filter(
        Device.user_id == user_id,
        EnergyReading.timestamp >= yesterday_start,
        EnergyReading.timestamp < yesterday_end
    ).all()
    
    yesterday_usage = sum(reading.power_usage for reading in yesterday_readings)
    
    # Calculate solar production today
    today_solar = SolarProduction.query.filter(
        SolarProduction.user_id == user_id,
        SolarProduction.timestamp >= today_start,
        SolarProduction.timestamp < today_end
    ).all()
    
    solar_production = sum(reading.production for reading in today_solar)
    
    # Get current battery level
    battery = BatteryStorage.query.filter_by(user_id=user_id).order_by(BatteryStorage.timestamp.desc()).first()
    battery_level = battery.level if battery else 0
    
    # Calculate cost based on current rate
    current_rate = get_current_electricity_rate()
    estimated_daily_cost = today_usage * current_rate
    
    # Calculate net usage (consumption minus production)
    net_usage = max(0, today_usage - solar_production)
    
    # Calculate carbon footprint estimation (0.4 kg CO2 per kWh is a rough average)
    carbon_footprint = net_usage * 0.4
    
    # Calculate day-over-day comparison
    if yesterday_usage > 0:
        usage_change_pct = ((today_usage - yesterday_usage) / yesterday_usage) * 100
    else:
        usage_change_pct = 0
    
    return {
        'today_usage': today_usage,
        'yesterday_usage': yesterday_usage,
        'solar_production': solar_production,
        'battery_level': battery_level,
        'estimated_daily_cost': estimated_daily_cost,
        'net_usage': net_usage,
        'carbon_footprint': carbon_footprint,
        'usage_change_pct': usage_change_pct,
        'current_rate': current_rate
    }

# def generate_charts_data(user_id):
#     # Get date ranges
#     end_date = datetime.utcnow()
#     start_date =

# dashboard.html

def generate_charts_data(user_id):
    # Get date ranges
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    
    # Get hourly energy usage for the past 7 days
    readings = EnergyReading.query.join(Device).filter(
        Device.user_id == user_id,
        EnergyReading.timestamp >= start_date,
        EnergyReading.timestamp <= end_date
    ).all()
    
    # Get solar production data
    solar_data = SolarProduction.query.filter(
        SolarProduction.user_id == user_id,
        SolarProduction.timestamp >= start_date,
        SolarProduction.timestamp <= end_date
    ).all()
    
    # Get battery level data
    battery_data = BatteryStorage.query.filter(
        BatteryStorage.user_id == user_id,
        BatteryStorage.timestamp >= start_date,
        BatteryStorage.timestamp <= end_date
    ).all()
    
    # Prepare data for charts
    # 1. Daily energy usage
    daily_usage = {}
    daily_solar = {}
    daily_net = {}
    
    for reading in readings:
        day = reading.timestamp.strftime('%Y-%m-%d')
        if day not in daily_usage:
            daily_usage[day] = 0
        daily_usage[day] += reading.power_usage
    
    for reading in solar_data:
        day = reading.timestamp.strftime('%Y-%m-%d')
        if day not in daily_solar:
            daily_solar[day] = 0
        daily_solar[day] += reading.production
    
    # Calculate net usage (grid consumption)
    for day in daily_usage:
        solar = daily_solar.get(day, 0)
        daily_net[day] = max(0, daily_usage[day] - solar)
    
    # 2. Hourly usage patterns
    hourly_usage = [0] * 24
    hourly_counts = [0] * 24
    
    for reading in readings:
        hour = reading.timestamp.hour
        hourly_usage[hour] += reading.power_usage
        hourly_counts[hour] += 1
    
    # Average the hourly usage
    for hour in range(24):
        if hourly_counts[hour] > 0:
            hourly_usage[hour] /= hourly_counts[hour]
    
    # 3. Device breakdown
    device_usage = {}
    devices = Device.query.filter_by(user_id=user_id).all()
    
    for device in devices:
        device_readings = [r for r in readings if r.device_id == device.id]
        device_usage[device.name] = sum(r.power_usage for r in device_readings)
    
    # 4. Battery level trend
    battery_trend = [(reading.timestamp.strftime('%Y-%m-%d %H:%M'), reading.level) 
                     for reading in battery_data]
    
    # Prepare data for json serialization in charts
    charts_data = {
        'daily_usage': {
            'labels': list(daily_usage.keys()),
            'consumption': list(daily_usage.values()),
            'solar': [daily_solar.get(day, 0) for day in daily_usage.keys()],
            'net': [daily_net.get(day, 0) for day in daily_usage.keys()]
        },
        'hourly_pattern': {
            'labels': list(range(24)),
            'values': hourly_usage
        },
        'device_breakdown': {
            'labels': list(device_usage.keys()),
            'values': list(device_usage.values())
        },
        'battery_trend': battery_trend
    }
    
    return charts_data

@app.route('/devices')
@login_required
def devices():
    devices = Device.query.filter_by(user_id=current_user.id).all()
    return render_template('devices.html', devices=devices)

@app.route('/devices/add', methods=['GET', 'POST'])
@login_required
def add_device():
    if request.method == 'POST':
        name = request.form.get('name')
        device_type = request.form.get('type')
        
        if not name or not device_type:
            flash('Please fill all fields', 'danger')
            return redirect(url_for('add_device'))
        
        device = Device(name=name, type=device_type, user_id=current_user.id)
        db.session.add(device)
        db.session.commit()
        
        flash('Device added successfully!', 'success')
        return redirect(url_for('devices'))
    
    return render_template('add_device.html')

@app.route('/devices/<int:device_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_device(device_id):
    device = Device.query.filter_by(id=device_id, user_id=current_user.id).first_or_404()
    
    if request.method == 'POST':
        name = request.form.get('name')
        device_type = request.form.get('type')
        
        if not name or not device_type:
            flash('Please fill all fields', 'danger')
            return redirect(url_for('edit_device', device_id=device_id))
        
        device.name = name
        device.type = device_type
        db.session.commit()
        
        flash('Device updated successfully!', 'success')
        return redirect(url_for('devices'))
    
    return render_template('edit_device.html', device=device)

@app.route('/devices/<int:device_id>/delete', methods=['POST'])
@login_required
def delete_device(device_id):
    device = Device.query.filter_by(id=device_id, user_id=current_user.id).first_or_404()
    
    db.session.delete(device)
    db.session.commit()
    
    flash('Device deleted successfully!', 'success')
    return redirect(url_for('devices'))

@app.route('/devices/<int:device_id>/toggle', methods=['POST'])
@login_required
def toggle_device(device_id):
    device = Device.query.filter_by(id=device_id, user_id=current_user.id).first_or_404()
    
    device.status = not device.status
    db.session.commit()
    
    return jsonify({'status': device.status})

@app.route('/recommendations')
@login_required
def recommendations():
    recommendations = Recommendation.query.filter_by(user_id=current_user.id).order_by(Recommendation.timestamp.desc()).all()
    return render_template('recommendations.html', recommendations=recommendations)

@app.route('/recommendations/<int:rec_id>/implement', methods=['POST'])
@login_required
def implement_recommendation(rec_id):
    recommendation = Recommendation.query.filter_by(id=rec_id, user_id=current_user.id).first_or_404()
    
    recommendation.implemented = True
    db.session.commit()
    
    flash('Recommendation marked as implemented!', 'success')
    return redirect(url_for('recommendations'))

@app.route('/alerts')
@login_required
def alerts():
    alerts = Alert.query.filter_by(user_id=current_user.id).order_by(Alert.timestamp.desc()).all()
    
    # Mark all as read
    for alert in alerts:
        if not alert.read:
            alert.read = True
    
    db.session.commit()
    
    return render_template('alerts.html', alerts=alerts)

@app.route('/automation')
@login_required
def automation():
    rules = AutomationRule.query.filter_by(user_id=current_user.id).all()
    devices = Device.query.filter_by(user_id=current_user.id).all()
    
    return render_template('automation.html', rules=rules, devices=devices)

@app.route('/automation/add', methods=['GET', 'POST'])
@login_required
def add_automation():
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        condition_type = request.form.get('condition_type')
        condition_value = request.form.get('condition_value')
        action_type = request.form.get('action_type')
        action_value = request.form.get('action_value', '')
        device_id = request.form.get('device_id')
        
        # Validate inputs
        if not name or not condition_type or not condition_value or not action_type or not device_id:
            flash('Please fill all required fields', 'danger')
            return redirect(url_for('add_automation'))
        
        # Validate device belongs to user
        device = Device.query.filter_by(id=device_id, user_id=current_user.id).first()
        if not device:
            flash('Invalid device selected', 'danger')
            return redirect(url_for('add_automation'))
        
        rule = AutomationRule(
            name=name,
            description=description,
            condition_type=condition_type,
            condition_value=condition_value,
            action_type=action_type,
            action_value=action_value,
            device_id=device_id,
            user_id=current_user.id
        )
        
        db.session.add(rule)
        db.session.commit()
        
        flash('Automation rule created successfully!', 'success')
        return redirect(url_for('automation'))
    
    devices = Device.query.filter_by(user_id=current_user.id).all()
    return render_template('add_automation.html', devices=devices)

@app.route('/automation/<int:rule_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_automation(rule_id):
    rule = AutomationRule.query.filter_by(id=rule_id, user_id=current_user.id).first_or_404()
    
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        condition_type = request.form.get('condition_type')
        condition_value = request.form.get('condition_value')
        action_type = request.form.get('action_type')
        action_value = request.form.get('action_value', '')
        device_id = request.form.get('device_id')
        
        # Validate inputs
        if not name or not condition_type or not condition_value or not action_type or not device_id:
            flash('Please fill all required fields', 'danger')
            return redirect(url_for('edit_automation', rule_id=rule_id))
        
        # Validate device belongs to user
        device = Device.query.filter_by(id=device_id, user_id=current_user.id).first()
        if not device:
            flash('Invalid device selected', 'danger')
            return redirect(url_for('edit_automation', rule_id=rule_id))
        
        rule.name = name
        rule.description = description
        rule.condition_type = condition_type
        rule.condition_value = condition_value
        rule.action_type = action_type
        rule.action_value = action_value
        rule.device_id = device_id
        
        db.session.commit()
        
        flash('Automation rule updated successfully!', 'success')
        return redirect(url_for('automation'))
    
    devices = Device.query.filter_by(user_id=current_user.id).all()
    return render_template('edit_automation.html', rule=rule, devices=devices)

@app.route('/automation/<int:rule_id>/toggle', methods=['POST'])
@login_required
def toggle_automation(rule_id):
    rule = AutomationRule.query.filter_by(id=rule_id, user_id=current_user.id).first_or_404()
    
    rule.enabled = not rule.enabled
    db.session.commit()
    
    return jsonify({'enabled': rule.enabled})

@app.route('/automation/<int:rule_id>/delete', methods=['POST'])
@login_required
def delete_automation(rule_id):
    rule = AutomationRule.query.filter_by(id=rule_id, user_id=current_user.id).first_or_404()
    
    db.session.delete(rule)
    db.session.commit()
    
    flash('Automation rule deleted successfully!', 'success')
    return redirect(url_for('automation'))

@app.route('/api/energy-data')
@login_required
def energy_data_api():
    # Get time range from request
    days = request.args.get('days', default=7, type=int)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Get energy readings
    readings = EnergyReading.query.join(Device).filter(
        Device.user_id == current_user.id,
        EnergyReading.timestamp >= start_date,
        EnergyReading.timestamp <= end_date
    ).all()
    
    # Get solar production data
    solar_data = SolarProduction.query.filter(
        SolarProduction.user_id == current_user.id,
        SolarProduction.timestamp >= start_date,
        SolarProduction.timestamp <= end_date
    ).all()
    
    # Prepare data by hour
    hourly_data = {}
    
    for reading in readings:
        hour_key = reading.timestamp.strftime('%Y-%m-%d %H:00')
        if hour_key not in hourly_data:
            hourly_data[hour_key] = {'consumption': 0, 'solar': 0}
        
        hourly_data[hour_key]['consumption'] += reading.power_usage
    
    for reading in solar_data:
        hour_key = reading.timestamp.strftime('%Y-%m-%d %H:00')
        if hour_key not in hourly_data:
            hourly_data[hour_key] = {'consumption': 0, 'solar': 0}
        
        hourly_data[hour_key]['solar'] += reading.production
    
    # Convert to list for JSON
    result = []
    for hour, data in sorted(hourly_data.items()):
        result.append({
            'timestamp': hour,
            'consumption': data['consumption'],
            'solar': data['solar'],
            'net': max(0, data['consumption'] - data['solar'])
        })
    
    return jsonify(result)

@app.route('/api/device-data')
@login_required
def device_data_api():
    # Get time range from request
    days = request.args.get('days', default=7, type=int)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Get all user devices
    devices = Device.query.filter_by(user_id=current_user.id).all()
    
    result = []
    for device in devices:
        # Get device readings
        readings = EnergyReading.query.filter(
            EnergyReading.device_id == device.id,
            EnergyReading.timestamp >= start_date,
            EnergyReading.timestamp <= end_date
        ).all()
        
        total_usage = sum(reading.power_usage for reading in readings)
        
        result.append({
            'device_id': device.id,
            'name': device.name,
            'type': device.type,
            'total_usage': total_usage,
            'status': device.status
        })
    
    return jsonify(result)

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    email = request.form.get('email')
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    
    # Validate current password
    if current_password and not current_user.check_password(current_password):
        flash('Current password is incorrect', 'danger')
        return redirect(url_for('profile'))
    
    # Update email if provided
    if email and email != current_user.email:
        if User.query.filter_by(email=email).first() and User.query.filter_by(email=email).first().id != current_user.id:
            flash('Email already in use', 'danger')
            return redirect(url_for('profile'))
        
        current_user.email = email
    
    # Update password if provided
    if new_password:
        current_user.set_password(new_password)
    
    db.session.commit()
    
    flash('Profile updated successfully!', 'success')
    return redirect(url_for('profile'))

# API endpoint for generating recommendations on demand
@app.route('/api/recommendations/generate', methods=['POST'])
@login_required
def generate_recommendations_api():
    if GEMINI_API_KEY:
        recommendations = generate_ai_recommendations(current_user.id)
    else:
        recommendations = generate_recommendations(current_user.id)
    
    return jsonify([{
        'message': rec['message'], 
        'impact': rec['impact']
    } for rec in recommendations])

# Create database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)