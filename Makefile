CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Werror -I./include
DEBUG_FLAGS = -g -O0
RELEASE_FLAGS = -O3 -DNDEBUG
SANITIZE_FLAGS = -fsanitize=address -fsanitize=undefined # sanitizers for memory errors and undefined behavior

# Directories
SRC_DIR = src
INC_DIR = include
TEST_DIR = tests
BUILD_DIR = build

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Test files
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJECTS = $(TEST_SOURCES:$(TEST_DIR)/%.cpp=$(BUILD_DIR)/test_%.o)

# Google test
GTEST_INCLUDE = -I/opt/homebrew/include -I/usr/local/include
GTEST_LIBS = -L/opt/homebrew/lib -L/usr/local/lib -lgtest -lgtest_main -pthread

# Executables
TEST_EXECUTABLE = $(BUILD_DIR)/all_tests

# Main targets
.PHONY: all clean debug release sanitize test test-all test-quick test-all-suites test-tensor test-matmul test-relu test-sigmoid test-softmax help install-gtest

all: debug

debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: $(TEST_EXECUTABLE)

release: CXXFLAGS += $(RELEASE_FLAGS)
release: $(TEST_EXECUTABLE)

sanitize: CXXFLAGS += $(DEBUG_FLAGS) $(SANITIZE_FLAGS)
sanitize: $(TEST_EXECUTABLE)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile test files
$(BUILD_DIR)/test_%.o: $(TEST_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(GTEST_INCLUDE) -c $< -o $@

# Build single test executable with all tests
$(TEST_EXECUTABLE): $(OBJECTS) $(TEST_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(GTEST_LIBS)

# Build tests only
test: $(TEST_EXECUTABLE)

# Run all tests
test-all-suites: $(TEST_EXECUTABLE)
	@echo "Running all test suites..."
	@./$(TEST_EXECUTABLE)

# Run specific test suites using gtest filters
test-tensor: $(TEST_EXECUTABLE)
	@echo "Running Tensor tests..."
	@./$(TEST_EXECUTABLE) --gtest_filter="TensorTest*"

test-matmul: $(TEST_EXECUTABLE)
	@echo "Running MatMul tests..."
	@./$(TEST_EXECUTABLE) --gtest_filter="MatmulTest*"

test-relu: $(TEST_EXECUTABLE)
	@echo "Running ReLU tests..."
	@./$(TEST_EXECUTABLE) --gtest_filter="ReluTest*"

test-sigmoid: $(TEST_EXECUTABLE)
	@echo "Running Sigmoid tests..."
	@./$(TEST_EXECUTABLE) --gtest_filter="SigmoidTest*"

test-softmax: $(TEST_EXECUTABLE)
	@echo "Running Softmax tests..."
	@./$(TEST_EXECUTABLE) --gtest_filter="SoftmaxTest*"

# Run comprehensive test suite
test-all:
	./run_tests.sh --all

# Run quick tests only
test-quick:
	./run_tests.sh --quick

# Clean build files
clean:
	rm -rf $(BUILD_DIR)

# Help targets
help:
	@echo "Available targets:"
	@echo "  all               Build all tests (default: debug)"
	@echo "  debug             Build debug version of test executable"
	@echo "  release           Build optimized release version"
	@echo "  sanitize          Build with memory sanitizers"
	@echo "  test              Build test executable"
	@echo "  test-all-suites   Run all test suites"
	@echo "  test-all          Run comprehensive test suite (via run_tests.sh)"
	@echo "  test-quick        Run quick debug tests only"
	@echo "  clean             Remove build files"
	@echo "  install-gtest     Show Google Test installation instructions"
	@echo ""
	@echo "Individual test targets (using gtest filters):"
	@echo "  test-tensor       Run only tensor class tests"
	@echo "  test-matmul       Run only matrix multiplication tests"
	@echo "  test-relu         Run only ReLU activation tests"
	@echo "  test-sigmoid      Run only sigmoid activation tests"
	@echo "  test-softmax      Run only softmax activation tests"
	@echo ""
	@echo "Test executable:"
	@echo "  $(TEST_EXECUTABLE)    - Single executable with all tests"
	@echo ""
	@echo "Examples:"
	@echo "  make test-relu                    # Run just ReLU tests"
	@echo "  $(TEST_EXECUTABLE) --gtest_list_tests  # List all available tests"

# Install google test (help target)
install-gtest:
	@echo "Please install Google Test:"
	@echo "macOS: brew install googletest"
	@echo "Ubuntu/Debian: sudo apt-get install libgtest-dev"
	@echo ""
	@echo "After installation, you can build and run tests with:"
	@echo "  make test-individual    # Run each test suite"
	@echo "  make test-all          # Run comprehensive test suite"