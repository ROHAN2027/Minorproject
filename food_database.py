"""Comprehensive Food Database for Diabetes Simulation Console.

Provides nutritional data (specifically carbohydrates) for foods commonly consumed
in Indian and Western diets, sourced from USDA FoodData Central and the Indian Food
Composition Tables (IFCT).

Used by the G2P2C simulation app for meal planning, carb estimation, and insulin
requirement calculations.
"""

from typing import Dict, List, Optional, Union, Any

# General weight and volume conversion constants
UNIT_CONVERSIONS: Dict[str, float] = {
    "oz": 28.35,            # 1 oz = 28.35 grams
    "pound": 453.6,         # 1 pound = 453.6 grams
    "cup": 240.0,           # 1 cup = 240 ml
    "tablespoon": 15.0,     # 1 tablespoon = 15 ml
    "teaspoon": 5.0,        # 1 teaspoon = 5 ml
    # Common aliases
    "ounces": 28.35,
    "pounds": 453.6,
    "cups": 240.0,
    "tablespoons": 15.0,
    "teaspoons": 5.0,
    "tbsp": 15.0,
    "tsp": 5.0,
}

# Comprehensive Food Database categorized by cuisine / food type
FOOD_DATABASE: Dict[str, List[Dict[str, Any]]] = {
    "Indian Breads": [
        {
            "name": "Roti/Chapati",
            "carbs_per_unit": 15.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 30.0,
            },
        },
        {
            "name": "Paratha",
            "carbs_per_unit": 30.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 80.0,
            },
        },
        {
            "name": "Naan",
            "carbs_per_unit": 45.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 90.0,
            },
        },
        {
            "name": "Puri",
            "carbs_per_unit": 12.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 25.0,
            },
        },
        {
            "name": "Bhatura",
            "carbs_per_unit": 40.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 100.0,
            },
        },
        {
            "name": "Dosa",
            "carbs_per_unit": 25.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 80.0,
            },
        },
        {
            "name": "Idli",
            "carbs_per_unit": 12.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 40.0,
            },
        },
        {
            "name": "Uttapam",
            "carbs_per_unit": 30.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 120.0,
            },
        },
        {
            "name": "Appam",
            "carbs_per_unit": 20.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 80.0,
            },
        },
        {
            "name": "Thepla",
            "carbs_per_unit": 18.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 40.0,
            },
        },
    ],
    "Indian Rice Dishes": [
        {
            "name": "Plain Cooked Rice",
            "carbs_per_unit": 45.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 180.0,
                "bowls": 1.5,
            },
        },
        {
            "name": "Biryani",
            "carbs_per_unit": 55.0,
            "default_unit": "plates",
            "default_qty": 1.0,
            "available_units": {
                "plates": 1.0,
                "grams": 1.0 / 250.0,
                "cups": 180.0 / 250.0,
            },
        },
        {
            "name": "Pulao",
            "carbs_per_unit": 40.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 180.0,
                "bowls": 1.5,
            },
        },
        {
            "name": "Khichdi",
            "carbs_per_unit": 35.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 200.0,
                "bowls": 1.5,
            },
        },
        {
            "name": "Lemon Rice",
            "carbs_per_unit": 42.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 180.0,
                "bowls": 1.5,
            },
        },
        {
            "name": "Curd Rice",
            "carbs_per_unit": 35.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 200.0,
                "bowls": 1.5,
            },
        },
        {
            "name": "Poha",
            "carbs_per_unit": 38.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 180.0,
                "plates": 250.0 / 180.0,
            },
        },
        {
            "name": "Upma",
            "carbs_per_unit": 30.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 200.0,
                "bowls": 1.5,
            },
        },
        {
            "name": "Jeera Rice",
            "carbs_per_unit": 40.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 180.0,
                "bowls": 1.5,
            },
        },
        {
            "name": "Fried Rice",
            "carbs_per_unit": 44.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 180.0,
                "plates": 250.0 / 180.0,
            },
        },
    ],
    "Indian Snacks": [
        {
            "name": "Samosa",
            "carbs_per_unit": 25.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 80.0,
            },
        },
        {
            "name": "Pakora/Bhaji",
            "carbs_per_unit": 8.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 30.0,
            },
        },
        {
            "name": "Vada",
            "carbs_per_unit": 15.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 50.0,
            },
        },
        {
            "name": "Pav Bhaji",
            "carbs_per_unit": 50.0,
            "default_unit": "plates",
            "default_qty": 1.0,
            "available_units": {
                "plates": 1.0,
            },
        },
        {
            "name": "Aloo Tikki",
            "carbs_per_unit": 18.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 60.0,
            },
        },
        {
            "name": "Chaat",
            "carbs_per_unit": 35.0,
            "default_unit": "plates",
            "default_qty": 1.0,
            "available_units": {
                "plates": 1.0,
                "grams": 1.0 / 150.0,
            },
        },
        {
            "name": "Dhokla",
            "carbs_per_unit": 10.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 40.0,
            },
        },
        {
            "name": "Kachori",
            "carbs_per_unit": 22.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 60.0,
            },
        },
        {
            "name": "Pani Puri",
            "carbs_per_unit": 30.0,
            "default_unit": "plates",
            "default_qty": 1.0,
            "available_units": {
                "plates": 1.0,
                "pieces": 1.0 / 6.0,
                "grams": 1.0 / 150.0,
            },
        },
        {
            "name": "Bhel Puri",
            "carbs_per_unit": 35.0,
            "default_unit": "plates",
            "default_qty": 1.0,
            "available_units": {
                "plates": 1.0,
                "grams": 1.0 / 150.0,
            },
        },
    ],
    "Indian Sweets": [
        {
            "name": "Gulab Jamun",
            "carbs_per_unit": 20.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 40.0,
            },
        },
        {
            "name": "Rasgulla",
            "carbs_per_unit": 18.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 40.0,
            },
        },
        {
            "name": "Jalebi",
            "carbs_per_unit": 22.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 30.0,
            },
        },
        {
            "name": "Laddu",
            "carbs_per_unit": 25.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 40.0,
            },
        },
        {
            "name": "Barfi",
            "carbs_per_unit": 18.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 30.0,
            },
        },
        {
            "name": "Halwa",
            "carbs_per_unit": 0.55,
            "default_unit": "grams",
            "default_qty": 100.0,
            "available_units": {
                "grams": 1.0,
                "cups": 200.0,
            },
        },
        {
            "name": "Kheer",
            "carbs_per_unit": 40.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 200.0,
                "bowls": 1.5,
            },
        },
        {
            "name": "Rasmalai",
            "carbs_per_unit": 22.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 50.0,
            },
        },
        {
            "name": "Mysore Pak",
            "carbs_per_unit": 20.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 35.0,
            },
        },
        {
            "name": "Sandesh",
            "carbs_per_unit": 12.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 35.0,
            },
        },
    ],
    "Fruits": [
        {
            "name": "Apple",
            "carbs_per_unit": 25.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 180.0,
                "oz": 28.35 / 180.0,
            },
        },
        {
            "name": "Banana",
            "carbs_per_unit": 27.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 120.0,
            },
        },
        {
            "name": "Mango",
            "carbs_per_unit": 25.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "pieces": 200.0 / 165.0,
                "grams": 1.0 / 165.0,
            },
        },
        {
            "name": "Orange",
            "carbs_per_unit": 15.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 130.0,
            },
        },
        {
            "name": "Grapes",
            "carbs_per_unit": 27.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 150.0,
            },
        },
        {
            "name": "Watermelon",
            "carbs_per_unit": 11.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 150.0,
            },
        },
        {
            "name": "Papaya",
            "carbs_per_unit": 14.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 145.0,
                "pieces": 300.0 / 145.0,
            },
        },
        {
            "name": "Chiku/Sapota",
            "carbs_per_unit": 15.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 75.0,
            },
        },
        {
            "name": "Pomegranate",
            "carbs_per_unit": 27.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 175.0,
                "pieces": 200.0 / 175.0,
            },
        },
        {
            "name": "Guava",
            "carbs_per_unit": 14.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 100.0,
            },
        },
    ],
    "Vegetables": [
        {
            "name": "Potato",
            "carbs_per_unit": 26.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 150.0,
            },
        },
        {
            "name": "Sweet Potato",
            "carbs_per_unit": 27.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 130.0,
            },
        },
        {
            "name": "Green Peas",
            "carbs_per_unit": 21.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 160.0,
            },
        },
        {
            "name": "Corn",
            "carbs_per_unit": 27.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "pieces": 100.0 / 145.0,
                "grams": 1.0 / 145.0,
            },
        },
        {
            "name": "Mixed Veg Curry",
            "carbs_per_unit": 15.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 200.0,
                "bowls": 1.5,
            },
        },
        {
            "name": "Dal/Lentils",
            "carbs_per_unit": 20.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 200.0,
                "bowls": 1.5,
            },
        },
        {
            "name": "Rajma/Kidney Beans",
            "carbs_per_unit": 22.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 180.0,
                "bowls": 1.5,
            },
        },
        {
            "name": "Chole/Chickpeas",
            "carbs_per_unit": 27.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 165.0,
                "bowls": 1.5,
            },
        },
        {
            "name": "Bhindi/Okra Masala",
            "carbs_per_unit": 10.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 160.0,
                "bowls": 1.5,
            },
        },
        {
            "name": "Palak Paneer",
            "carbs_per_unit": 8.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 200.0,
                "bowls": 1.5,
            },
        },
    ],
    "Dairy": [
        {
            "name": "Milk",
            "carbs_per_unit": 12.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "ml": 1.0 / 240.0,
                "glasses": 250.0 / 240.0,
            },
        },
        {
            "name": "Curd/Yogurt",
            "carbs_per_unit": 11.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 245.0,
                "bowls": 1.5,
            },
        },
        {
            "name": "Lassi (Sweet)",
            "carbs_per_unit": 25.0,
            "default_unit": "glasses",
            "default_qty": 1.0,
            "available_units": {
                "glasses": 1.0,
                "ml": 1.0 / 250.0,
            },
        },
        {
            "name": "Buttermilk/Chaas",
            "carbs_per_unit": 5.0,
            "default_unit": "glasses",
            "default_qty": 1.0,
            "available_units": {
                "glasses": 1.0,
                "ml": 1.0 / 250.0,
            },
        },
        {
            "name": "Paneer",
            "carbs_per_unit": 0.03,
            "default_unit": "grams",
            "default_qty": 100.0,
            "available_units": {
                "grams": 1.0,
                "pieces": 30.0,
            },
        },
        {
            "name": "Ice Cream",
            "carbs_per_unit": 17.0,
            "default_unit": "scoops",
            "default_qty": 1.0,
            "available_units": {
                "scoops": 1.0,
                "grams": 1.0 / 65.0,
                "cups": 130.0 / 65.0,
            },
        },
        {
            "name": "Cheese",
            "carbs_per_unit": 1.0,
            "default_unit": "slices",
            "default_qty": 1.0,
            "available_units": {
                "slices": 1.0,
                "grams": 1.0 / 28.0,
            },
        },
        {
            "name": "Greek Yogurt",
            "carbs_per_unit": 8.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 200.0,
            },
        },
        {
            "name": "Soy Milk",
            "carbs_per_unit": 8.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "ml": 1.0 / 240.0,
                "glasses": 250.0 / 240.0,
            },
        },
        {
            "name": "Almond Milk (Unsweetened)",
            "carbs_per_unit": 2.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "ml": 1.0 / 240.0,
                "glasses": 250.0 / 240.0,
            },
        },
    ],
    "Beverages": [
        {
            "name": "Tea with Sugar",
            "carbs_per_unit": 8.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
            },
        },
        {
            "name": "Coffee with Sugar",
            "carbs_per_unit": 8.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
            },
        },
        {
            "name": "Mango Juice",
            "carbs_per_unit": 30.0,
            "default_unit": "glasses",
            "default_qty": 1.0,
            "available_units": {
                "glasses": 1.0,
                "ml": 1.0 / 250.0,
            },
        },
        {
            "name": "Orange Juice",
            "carbs_per_unit": 26.0,
            "default_unit": "glasses",
            "default_qty": 1.0,
            "available_units": {
                "glasses": 1.0,
                "ml": 1.0 / 250.0,
            },
        },
        {
            "name": "Coconut Water",
            "carbs_per_unit": 9.0,
            "default_unit": "glasses",
            "default_qty": 1.0,
            "available_units": {
                "glasses": 1.0,
                "ml": 1.0 / 240.0,
            },
        },
        {
            "name": "Sugarcane Juice",
            "carbs_per_unit": 40.0,
            "default_unit": "glasses",
            "default_qty": 1.0,
            "available_units": {
                "glasses": 1.0,
                "ml": 1.0 / 250.0,
            },
        },
        {
            "name": "Cola/Soft Drink",
            "carbs_per_unit": 35.0,
            "default_unit": "cans",
            "default_qty": 1.0,
            "available_units": {
                "cans": 1.0,
                "ml": 1.0 / 330.0,
                "glasses": 250.0 / 330.0,
            },
        },
        {
            "name": "Nimbu Pani/Lemonade",
            "carbs_per_unit": 15.0,
            "default_unit": "glasses",
            "default_qty": 1.0,
            "available_units": {
                "glasses": 1.0,
                "ml": 1.0 / 250.0,
            },
        },
        {
            "name": "Green Tea (Unsweetened)",
            "carbs_per_unit": 0.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "ml": 1.0 / 240.0,
            },
        },
        {
            "name": "Apple Juice",
            "carbs_per_unit": 28.0,
            "default_unit": "glasses",
            "default_qty": 1.0,
            "available_units": {
                "glasses": 1.0,
                "ml": 1.0 / 250.0,
            },
        },
    ],
    "Cereals & Grains": [
        {
            "name": "Oats",
            "carbs_per_unit": 27.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 240.0,
            },
        },
        {
            "name": "Cornflakes",
            "carbs_per_unit": 24.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 30.0,
            },
        },
        {
            "name": "Muesli",
            "carbs_per_unit": 40.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 55.0,
            },
        },
        {
            "name": "Bread (White)",
            "carbs_per_unit": 14.0,
            "default_unit": "slices",
            "default_qty": 1.0,
            "available_units": {
                "slices": 1.0,
                "grams": 1.0 / 30.0,
            },
        },
        {
            "name": "Bread (Brown/Wheat)",
            "carbs_per_unit": 12.0,
            "default_unit": "slices",
            "default_qty": 1.0,
            "available_units": {
                "slices": 1.0,
                "grams": 1.0 / 30.0,
            },
        },
        {
            "name": "Pasta",
            "carbs_per_unit": 43.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 140.0,
            },
        },
        {
            "name": "Maggi Noodles",
            "carbs_per_unit": 45.0,
            "default_unit": "packs",
            "default_qty": 1.0,
            "available_units": {
                "packs": 1.0,
                "grams": 1.0 / 70.0,
            },
        },
        {
            "name": "Biscuit",
            "carbs_per_unit": 5.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 8.0,
                "packs": 10.0,
            },
        },
        {
            "name": "Brown Rice (Cooked)",
            "carbs_per_unit": 45.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 195.0,
            },
        },
        {
            "name": "Quinoa (Cooked)",
            "carbs_per_unit": 39.0,
            "default_unit": "cups",
            "default_qty": 1.0,
            "available_units": {
                "cups": 1.0,
                "grams": 1.0 / 185.0,
            },
        },
    ],
    "Western/Fast Food": [
        {
            "name": "Pizza",
            "carbs_per_unit": 30.0,
            "default_unit": "slices",
            "default_qty": 1.0,
            "available_units": {
                "slices": 1.0,
                "grams": 1.0 / 110.0,
            },
        },
        {
            "name": "Burger",
            "carbs_per_unit": 35.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
            },
        },
        {
            "name": "French Fries",
            "carbs_per_unit": 40.0,
            "default_unit": "servings",
            "default_qty": 1.0,
            "available_units": {
                "servings": 1.0,
                "grams": 1.0 / 115.0,
            },
        },
        {
            "name": "Sandwich",
            "carbs_per_unit": 30.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
            },
        },
        {
            "name": "Pancake",
            "carbs_per_unit": 22.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 75.0,
            },
        },
        {
            "name": "Wrap/Roll",
            "carbs_per_unit": 28.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
            },
        },
        {
            "name": "Hot Dog",
            "carbs_per_unit": 22.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
            },
        },
        {
            "name": "Fried Chicken",
            "carbs_per_unit": 8.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 100.0,
            },
        },
        {
            "name": "Chicken Nuggets",
            "carbs_per_unit": 16.0,
            "default_unit": "servings",
            "default_qty": 1.0,
            "available_units": {
                "servings": 1.0,
                "pieces": 1.0 / 6.0,
                "grams": 1.0 / 100.0,
            },
        },
        {
            "name": "Garlic Bread",
            "carbs_per_unit": 15.0,
            "default_unit": "pieces",
            "default_qty": 1.0,
            "available_units": {
                "pieces": 1.0,
                "grams": 1.0 / 30.0,
            },
        },
    ],
    "Custom": [
        {
            "name": "Custom Carbohydrates (1g/g)",
            "carbs_per_unit": 1.0,
            "default_unit": "grams",
            "default_qty": 15.0,
            "available_units": {
                "grams": 1.0,
                "oz": 28.35,
            },
        },
        {
            "name": "Custom Snack (15g carbs)",
            "carbs_per_unit": 15.0,
            "default_unit": "servings",
            "default_qty": 1.0,
            "available_units": {
                "servings": 1.0,
            },
        },
        {
            "name": "Custom Snack (20g carbs)",
            "carbs_per_unit": 20.0,
            "default_unit": "servings",
            "default_qty": 1.0,
            "available_units": {
                "servings": 1.0,
            },
        },
        {
            "name": "Custom Meal (30g carbs)",
            "carbs_per_unit": 30.0,
            "default_unit": "servings",
            "default_qty": 1.0,
            "available_units": {
                "servings": 1.0,
            },
        },
        {
            "name": "Custom Meal (45g carbs)",
            "carbs_per_unit": 45.0,
            "default_unit": "servings",
            "default_qty": 1.0,
            "available_units": {
                "servings": 1.0,
            },
        },
        {
            "name": "Custom Meal (60g carbs)",
            "carbs_per_unit": 60.0,
            "default_unit": "servings",
            "default_qty": 1.0,
            "available_units": {
                "servings": 1.0,
            },
        },
        {
            "name": "Custom Meal (75g carbs)",
            "carbs_per_unit": 75.0,
            "default_unit": "servings",
            "default_qty": 1.0,
            "available_units": {
                "servings": 1.0,
            },
        },
        {
            "name": "Custom Drink (25g carbs)",
            "carbs_per_unit": 25.0,
            "default_unit": "glasses",
            "default_qty": 1.0,
            "available_units": {
                "glasses": 1.0,
                "ml": 1.0 / 250.0,
            },
        },
    ],
}


def get_food_list() -> List[str]:
    """Return a flat list of all food names across all categories, sorted alphabetically.

    Returns:
        List[str]: Alphabetically sorted food names.
    """
    food_names = set()
    for category_foods in FOOD_DATABASE.values():
        for food in category_foods:
            food_names.add(food["name"])
    return sorted(list(food_names))


def get_food_info(food_name: str) -> Optional[Dict[str, Any]]:
    """Retrieve the nutritional dictionary for a given food name.

    Performs an exact match search first, followed by a case-insensitive,
    trimmed search if no exact match is found.

    Args:
        food_name: The name of the food item to search for.

    Returns:
        Optional[Dict[str, Any]]: The food item dictionary, or None if not found.
    """
    if not food_name or not isinstance(food_name, str):
        return None

    target = food_name.strip()
    target_lower = target.lower()

    # Exact match first
    for category_foods in FOOD_DATABASE.values():
        for food in category_foods:
            if food["name"] == target:
                return food

    # Case-insensitive match fallback
    for category_foods in FOOD_DATABASE.values():
        for food in category_foods:
            if food["name"].strip().lower() == target_lower:
                return food

    return None


def calculate_carbs(food_name: str, quantity: float, unit: str) -> float:
    """Calculate total carbohydrates for a given food, quantity, and unit.

    Multiplies carbs_per_unit by quantity and by the unit conversion factor
    relative to the default unit.

    Args:
        food_name: The name of the food item.
        quantity: The quantity/amount of the food.
        unit: The unit of measurement (e.g., 'pieces', 'cups', 'grams').

    Returns:
        float: Total carbohydrates in grams, rounded to 2 decimal places.
               Returns 0.0 if food is not found.
    """
    food = get_food_info(food_name)
    if not food:
        return 0.0

    available_units = food.get("available_units", {})

    # Check for direct unit match
    if unit in available_units:
        conversion_factor = available_units[unit]
    else:
        # Fallback to normalized / case-insensitive unit match
        unit_clean = unit.strip().lower()
        conversion_factor = None

        for u_name, u_factor in available_units.items():
            u_clean = u_name.strip().lower()
            if u_clean == unit_clean:
                conversion_factor = u_factor
                break
            # Handle singular/plural differences (e.g. 'cup' vs 'cups')
            if u_clean.rstrip("s") == unit_clean.rstrip("s"):
                conversion_factor = u_factor
                break
            # Handle 'g' vs 'grams' / 'gram'
            if unit_clean in ("g", "gram", "grams") and u_clean in ("g", "gram", "grams"):
                conversion_factor = u_factor
                break

        if conversion_factor is None:
            # Default to 1.0 if unit cannot be resolved
            conversion_factor = 1.0

    total_carbs = food["carbs_per_unit"] * quantity * conversion_factor
    return float(round(total_carbs, 2))


def get_food_categories() -> List[str]:
    """Return a list of all available food categories.

    Returns:
        List[str]: List of category names.
    """
    return list(FOOD_DATABASE.keys())


def get_foods_by_category(category: str) -> List[Dict[str, Any]]:
    """Return all food items in a given category.

    Args:
        category: Name of category.

    Returns:
        List[Dict[str, Any]]: List of food dictionaries in category.
    """
    return FOOD_DATABASE.get(category, [])
