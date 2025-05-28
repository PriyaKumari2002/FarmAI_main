def get_disease_alerts(temperature, humidity):
    """
    Generate plant disease alerts based on temperature and humidity conditions.
    
    Args:
        temperature (float): Current temperature in Celsius
        humidity (float): Current humidity percentage
        
    Returns:
        list: List of alert messages with recommendations
    """
    alerts = []
    
    # High temperature and high humidity conditions
    if temperature > 30 and humidity > 80:
        alerts.extend([
            "⚠️ High risk of fungal diseases (powdery mildew, leaf spot, anthracnose)",
            "🚨 Bacterial wilt and blight likely in these conditions",
            "💡 Recommendations: Apply fungicides, improve air circulation, avoid overhead watering"
        ])
    
    # Hot and dry conditions
    elif temperature > 28 and humidity < 40:
        alerts.extend([
            "🌞 Extreme heat and dry conditions detected",
            "🕷️ Spider mites and thrips thrive in these conditions",
            "🔥 Plants may experience heat stress and wilting",
            "💡 Recommendations: Increase irrigation, use shade cloth, apply miticides if needed"
        ])
    
    # Cool and humid conditions
    elif temperature < 20 and humidity > 70:
        alerts.extend([
            "⚠️ Cool and damp conditions ideal for fungal growth",
            "🍄 Watch for downy mildew, gray mold (Botrytis), and late blight",
            "💧 Root rot diseases may develop in waterlogged soils",
            "💡 Recommendations: Reduce watering, improve drainage, apply protective fungicides"
        ])
    
    # Moderate temperature with high humidity
    elif 20 <= temperature <= 28 and humidity > 75:
        alerts.extend([
            "⚠️ Moderate risk of foliar diseases",
            "🦠 Bacterial leaf spot and fungal rust possible",
            "💡 Recommendations: Water in morning, space plants properly, remove affected leaves"
        ])
    
    # No significant risks
    else:
        alerts.append("✅ Conditions are generally favorable with no significant disease risks detected")
    
    # Add general recommendations for all conditions
    alerts.append("\n🌱 General best practice: Monitor plants regularly and maintain good sanitation")
    
    return alerts
 