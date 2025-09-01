#!/bin/bash

# run_tests.sh - Comprehensive test runner for miniNN project
# Usage: ./run_tests.sh [options]
# Options:
#   --quick      : Run only basic tests (debug mode)
#   --sanitize   : Run with memory sanitizers
#   --release    : Run optimized release build
#   --individual : Run individual test suites separately
#   --all        : Run all test configurations (default)
#   --help       : Show this help message

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to run tests with a specific configuration
run_test_config() {
    local config=$1
    local description=$2
    local individual=$3
    
    print_header "Running $description"
    
    # Clean previous build
    make clean > /dev/null 2>&1
    
    # Build with specified configuration
    if make $config > /dev/null 2>&1; then
        print_success "Build successful ($config)"
    else
        print_error "Build failed ($config)"
        return 1
    fi
    
    # Run tests - either individual or combined
    if [ "$individual" = "true" ]; then
        if run_individual_tests; then
            print_success "All individual tests passed ($config)"
            return 0
        else
            print_error "Some individual tests failed ($config)"
            return 1
        fi
    else
        # Run all tests together
        if ./build/all_tests > /dev/null 2>&1; then
            print_success "All tests passed ($config)"
            return 0
        else
            print_error "Tests failed ($config)"
            return 1
        fi
    fi
}

# Function to run individual test suites
run_individual_tests() {
    local failed=0
    local test_filters=("TensorTest*" "MatmulTest*" "ReluTest*" "SigmoidTest*" "SoftmaxTest*")
    local test_names=("Tensor" "MatMul" "ReLU" "Sigmoid" "Softmax")
    local exe="./build/all_tests"
    
    if [ ! -f "$exe" ]; then
        print_error "  Test executable not found: $exe"
        return 1
    fi
    
    for i in "${!test_filters[@]}"; do
        local filter="${test_filters[$i]}"
        local name="${test_names[$i]}"
        
        echo "  Running $name tests..."
        if $exe --gtest_filter="$filter" > /dev/null 2>&1; then
            print_success "  $name tests passed"
        else
            print_error "  $name tests failed"
            failed=1
        fi
    done
    
    return $failed
}

# Function to run valgrind if available
run_valgrind() {
    if command -v valgrind &> /dev/null; then
        print_header "Running Valgrind Memory Check"
        # Build debug version for valgrind
        make clean > /dev/null 2>&1
        make debug > /dev/null 2>&1
        
        if valgrind --leak-check=full --error-exitcode=1 --quiet ./build/all_tests > /dev/null 2>&1; then
            print_success "Valgrind: No memory leaks detected"
        else
            print_warning "Valgrind: Potential issues detected (run manually for details)"
            return 1
        fi
    else
        print_warning "Valgrind not available, skipping memory leak check"
    fi
}

# Function to show help
show_help() {
    echo "miniNN Test Runner"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --quick       Run only debug tests (fastest)"
    echo "  --sanitize    Run with memory sanitizers"
    echo "  --release     Run optimized release build"
    echo "  --individual  Run individual test suites separately"
    echo "  --valgrind    Run with valgrind memory check"
    echo "  --all         Run all test configurations (default)"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                         # Run all configurations"
    echo "  $0 --quick                 # Quick debug test only"
    echo "  $0 --individual            # Run individual test suites"
    echo "  $0 --sanitize --individual # Sanitized build with individual tests"
    echo "  $0 --release --valgrind    # Release build + valgrind"
}

# Parse command line arguments
QUICK=false
SANITIZE=false
RELEASE=false
INDIVIDUAL=false
VALGRIND=false
ALL=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK=true
            ALL=false
            shift
            ;;
        --sanitize)
            SANITIZE=true
            ALL=false
            shift
            ;;
        --release)
            RELEASE=true
            ALL=false
            shift
            ;;
        --individual)
            INDIVIDUAL=true
            shift
            ;;
        --valgrind)
            VALGRIND=true
            shift
            ;;
        --all)
            ALL=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
echo -e "${BLUE}miniNN Test Runner${NC}"
echo -e "${BLUE}==================${NC}"

# Check if Google Test is available
if ! pkg-config --exists gtest 2>/dev/null && ! find /opt/homebrew /usr/local -name "libgtest*" 2>/dev/null | grep -q .; then
    print_warning "Google Test might not be properly installed"
    echo "Install with: brew install googletest (macOS) or sudo apt-get install libgtest-dev (Ubuntu)"
fi

TOTAL_PASSED=0
TOTAL_FAILED=0

# Run tests based on arguments
if $ALL || $QUICK; then
    if run_test_config "debug" "Debug Build Tests" "$INDIVIDUAL"; then
        ((TOTAL_PASSED++))
    else
        ((TOTAL_FAILED++))
    fi
fi

if $ALL || $SANITIZE; then
    if run_test_config "sanitize" "Sanitized Build Tests (Memory Safety)" "$INDIVIDUAL"; then
        ((TOTAL_PASSED++))
    else
        ((TOTAL_FAILED++))
    fi
fi

if $ALL || $RELEASE; then
    if run_test_config "release" "Release Build Tests (Optimized)" "$INDIVIDUAL"; then
        ((TOTAL_PASSED++))
    else
        ((TOTAL_FAILED++))
    fi
fi

if $VALGRIND; then
    if run_valgrind; then
        ((TOTAL_PASSED++))
    else
        ((TOTAL_FAILED++))
    fi
fi

# Summary
print_header "Test Summary"
echo -e "Configurations passed: ${GREEN}$TOTAL_PASSED${NC}"
echo -e "Configurations failed: ${RED}$TOTAL_FAILED${NC}"

if [ $TOTAL_FAILED -eq 0 ]; then
    print_success "All test configurations passed."
    exit 0
else
    print_error "Some test configurations failed."
    exit 1
fi
