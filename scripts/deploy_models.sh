#!/bin/bash
# Model Deployment Script for BrowerAI
# Automates model generation, validation, and deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models/local"

echo "=========================================="
echo "BrowerAI Model Deployment Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python availability
check_python() {
    if command -v python3 &> /dev/null; then
        echo -e "${GREEN}✓${NC} Python3 found: $(python3 --version)"
        return 0
    else
        echo -e "${RED}✗${NC} Python3 not found"
        return 1
    fi
}

# Check required Python packages
check_python_packages() {
    echo ""
    echo "Checking Python packages..."
    
    local required_packages=("numpy" "onnx")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} $package installed"
        else
            echo -e "${YELLOW}!${NC} $package not found"
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}Installing missing packages...${NC}"
        pip3 install "${missing_packages[@]}" || {
            echo -e "${RED}✗${NC} Failed to install packages"
            return 1
        }
    fi
    
    return 0
}

# Generate minimal demo models
generate_minimal_models() {
    echo ""
    echo "=========================================="
    echo "Generating Minimal Demo Models"
    echo "=========================================="
    
    python3 "$SCRIPT_DIR/generate_minimal_models.py" --output-dir "$MODELS_DIR"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Models generated successfully"
        return 0
    else
        echo -e "${RED}✗${NC} Model generation failed"
        return 1
    fi
}

# Validate generated models
validate_models() {
    echo ""
    echo "=========================================="
    echo "Validating Models"
    echo "=========================================="
    
    local model_count=0
    local valid_count=0
    
    for model_file in "$MODELS_DIR"/*.onnx; do
        if [ -f "$model_file" ]; then
            model_count=$((model_count + 1))
            filename=$(basename "$model_file")
            
            # Check file size
            size=$(stat -f%z "$model_file" 2>/dev/null || stat -c%s "$model_file" 2>/dev/null)
            size_kb=$((size / 1024))
            
            if [ $size -gt 100 ]; then
                echo -e "${GREEN}✓${NC} $filename (${size_kb}KB)"
                valid_count=$((valid_count + 1))
            else
                echo -e "${RED}✗${NC} $filename (too small: ${size_kb}KB)"
            fi
        fi
    done
    
    echo ""
    echo "Valid models: $valid_count / $model_count"
    
    if [ $valid_count -gt 0 ]; then
        return 0
    else
        return 1
    fi
}

# List deployed models
list_models() {
    echo ""
    echo "=========================================="
    echo "Deployed Models"
    echo "=========================================="
    
    if [ ! -d "$MODELS_DIR" ]; then
        echo -e "${YELLOW}No models directory found${NC}"
        return
    fi
    
    local count=0
    for model_file in "$MODELS_DIR"/*.onnx; do
        if [ -f "$model_file" ]; then
            count=$((count + 1))
            filename=$(basename "$model_file")
            size=$(stat -f%z "$model_file" 2>/dev/null || stat -c%s "$model_file" 2>/dev/null)
            size_kb=$((size / 1024))
            
            echo "  $count. $filename (${size_kb}KB)"
        fi
    done
    
    if [ $count -eq 0 ]; then
        echo -e "${YELLOW}No ONNX models found${NC}"
    else
        echo ""
        echo -e "${GREEN}Total: $count model(s)${NC}"
    fi
}

# Check model config
check_model_config() {
    echo ""
    echo "=========================================="
    echo "Model Configuration"
    echo "=========================================="
    
    local config_file="$PROJECT_ROOT/models/model_config.toml"
    
    if [ -f "$config_file" ]; then
        echo -e "${GREEN}✓${NC} Configuration file found"
        
        # Count configured models
        local configured=$(grep -c "^\[\[models\]\]" "$config_file" || echo "0")
        echo "  Configured models: $configured"
        
        # Show enabled models
        echo ""
        echo "Enabled models:"
        grep "^name = " "$config_file" | head -5
    else
        echo -e "${RED}✗${NC} Configuration file not found"
    fi
}

# Main deployment workflow
main() {
    echo "Project root: $PROJECT_ROOT"
    echo "Models directory: $MODELS_DIR"
    echo ""
    
    # Step 1: Check Python
    if ! check_python; then
        echo -e "${RED}Deployment aborted: Python3 required${NC}"
        exit 1
    fi
    
    # Step 2: Check Python packages
    if ! check_python_packages; then
        echo -e "${YELLOW}Warning: Some packages missing, continuing anyway...${NC}"
    fi
    
    # Step 3: Generate models
    generate_minimal_models
    
    # Step 4: Validate models
    validate_models
    
    # Step 5: List deployed models
    list_models
    
    # Step 6: Check configuration
    check_model_config
    
    # Final instructions
    echo ""
    echo "=========================================="
    echo "Next Steps"
    echo "=========================================="
    echo "1. Build with AI support:"
    echo "   cargo build --features ai"
    echo ""
    echo "2. Run demos:"
    echo "   cargo run --features ai"
    echo ""
    echo "3. Test models:"
    echo "   cargo test --features ai"
    echo ""
    echo -e "${GREEN}✓ Deployment complete!${NC}"
}

# Run main workflow
main
