# 📋 Input Validation Guide - Crop Yield Prediction

## Overview
All numerical inputs are validated with minimum and maximum ranges to ensure data quality and model accuracy. Invalid inputs are rejected with clear error messages.

---

## ✅ Validated Input Ranges

### Environmental Factors

#### Temperature (°C)
- **Range**: 8 - 43°C
- **Type**: Decimal number
- **Step**: 0.1
- **Example**: 26.5
- **Validation**: ❌ Rejects values < 8 or > 43

#### Humidity (%)
- **Range**: 20 - 99%
- **Type**: Decimal number
- **Step**: 0.1
- **Example**: 65.5
- **Validation**: ❌ Rejects values < 20 or > 99

#### Moisture
- **Range**: 10 - 82
- **Type**: Decimal number
- **Step**: 0.1
- **Example**: 45.3
- **Validation**: ❌ Rejects values < 10 or > 82

---

### Soil Nutrients

#### Nitrogen (N)
- **Range**: 0 - 140 kg/hectare
- **Type**: Decimal number (non-negative)
- **Step**: 0.1
- **Example**: 37.0
- **Validation**: ❌ Rejects negative values or > 140

#### Phosphorous (P)
- **Range**: 0 - 144 kg/hectare
- **Type**: Decimal number (non-negative)
- **Step**: 0.1
- **Example**: 36.0
- **Validation**: ❌ Rejects negative values or > 144

#### Potassium (K)
- **Range**: 0 - 205 kg/hectare
- **Type**: Decimal number (non-negative)
- **Step**: 0.1
- **Example**: 0.0
- **Validation**: ❌ Rejects negative values or > 205

---

### Categorical Inputs

#### Soil Type
- **Options**: Black, Clayey, Loamy, Red, Sandy
- **Type**: Dropdown selection
- **Required**: Yes
- **Validation**: ❌ Must select one option

#### Crop Type
- **Options**: Barley, Cotton, Ground Nuts, Maize, Millets, Oil seeds, Paddy, Pulses, Sugarcane, Tobacco, Wheat
- **Type**: Dropdown selection
- **Required**: Yes
- **Validation**: ❌ Must select one option

#### Fertilizer Name
- **Options**: 10-26-26, 14-35-14, 17-17-17, 20-20, 28-28, DAP, Urea
- **Type**: Dropdown selection
- **Required**: Yes
- **Validation**: ❌ Must select one option

---

## 🔴 Error Messages

### Out-of-Range Errors
```
Temperature must be between 8 and 43
Humidity must be between 20 and 99
Moisture must be between 10 and 82
Nitrogen must be between 0 and 140
Phosphorous must be between 0 and 144
Potassium must be between 0 and 205
```

### Invalid Input Errors
```
Temperature must be a valid number
Humidity must be a valid number
Moisture must be a valid number
Nitrogen must be a valid number
Phosphorous must be a valid number
Potassium must be a valid number
```

### Missing Selection Errors
```
Please select a Soil Type
Please select a Crop Type
Please select a Fertilizer
```

---

## 🎯 Validation Features

### Real-Time Validation
- ✅ Validates as you type
- ✅ Red border indicates invalid input
- ✅ Tooltip shows valid range
- ✅ Automatic correction on blur

### Form Submission Validation
- ✅ All fields checked before sending
- ✅ Multiple errors combined in one message
- ✅ Clear error display in red box
- ✅ Fields highlighted for correction

### Browser-Level Validation
- ✅ HTML5 min/max attributes
- ✅ Number input type prevents non-numeric
- ✅ Step attribute for decimal precision
- ✅ Required attribute enforces completion

---

## 📊 Example Valid Inputs

### Example 1: Maize Crop
```
Temperature: 26°C
Humidity: 52%
Moisture: 38
Nitrogen: 37 kg/hectare
Phosphorous: 0 kg/hectare
Potassium: 0 kg/hectare
Soil Type: Sandy
Crop Type: Maize
Fertilizer: Urea
```
✅ **All inputs valid** → Prediction: ~4,500 kg/hectare

### Example 2: Sugarcane Crop
```
Temperature: 29°C
Humidity: 52%
Moisture: 45
Nitrogen: 12 kg/hectare
Phosphorous: 36 kg/hectare
Potassium: 0 kg/hectare
Soil Type: Loamy
Crop Type: Sugarcane
Fertilizer: DAP
```
✅ **All inputs valid** → Prediction: ~7,200 kg/hectare

### Example 3: Cotton Crop
```
Temperature: 34°C
Humidity: 65%
Moisture: 60
Nitrogen: 7 kg/hectare
Phosphorous: 30 kg/hectare
Potassium: 9 kg/hectare
Soil Type: Black
Crop Type: Cotton
Fertilizer: 14-35-14
```
✅ **All inputs valid** → Prediction: ~5,000 kg/hectare

---

## ❌ Example Invalid Inputs

### Invalid Example 1: Out-of-Range Temperature
```
Temperature: 50°C  ❌ (Max: 43°C)
Error: "Temperature must be between 8 and 43"
```

### Invalid Example 2: Negative Nitrogen
```
Nitrogen: -10 kg/hectare  ❌ (Min: 0)
Error: "Nitrogen must be between 0 and 140"
```

### Invalid Example 3: Missing Crop Type
```
Crop Type: (not selected)  ❌
Error: "Please select a Crop Type"
```

### Invalid Example 4: Non-Numeric Input
```
Humidity: "high"  ❌ (Must be number)
Error: "Humidity must be a valid number"
```

---

## 💡 Tips for Using the Form

1. **Use Range Hints**: Look at the range displayed next to each label
2. **Check Tooltips**: Hover over fields to see valid ranges
3. **Watch for Red Borders**: Red border = invalid input
4. **Read Error Messages**: Clear messages explain what's wrong
5. **Use Dropdowns**: Don't type in categorical fields
6. **Decimal Values**: Use decimal point (.) not comma (,)
7. **No Negative Values**: Nutrients can't be negative

---

## 🔧 Technical Details

### Validation Layers

**Layer 1: HTML5 Attributes**
```html
<input type="number" min="8" max="43" step="0.1" required>
```

**Layer 2: JavaScript Real-Time**
```javascript
if (value < range.min || value > range.max) {
    input.style.borderColor = '#f44';
}
```

**Layer 3: Form Submission**
```javascript
// Validate all fields before sending to server
if (!isValid) {
    showErrorMessage(errorMessages);
    return;
}
```

**Layer 4: Server-Side** (Backend validation in Flask)
```python
# Additional validation on server for security
```

---

## 📱 Mobile Compatibility

- ✅ Touch-friendly input fields
- ✅ Mobile number keyboard for numerical inputs
- ✅ Dropdown menus work on all devices
- ✅ Error messages display clearly
- ✅ Responsive layout adjusts to screen size

---

## 🆘 Troubleshooting

### Problem: "Field shows red border but value looks correct"
**Solution**: Check that value is within the displayed range. Use decimal point (.) not comma (,).

### Problem: "Can't enter decimal values"
**Solution**: Make sure you're using a decimal point (.). The step is set to 0.1 for precision.

### Problem: "Dropdown not showing options"
**Solution**: Click on the dropdown arrow. All options should appear in a list.

### Problem: "Error message says field is required but I filled it"
**Solution**: Make sure you selected an option from the dropdown (not just typed).

---

**Last Updated**: 2025-10-29
**Status**: ✅ All validation working correctly

