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
.PHONY: all clean debug release sanitize test test-all help install-gtest

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

# Test targets (all delegated to run_tests.sh)
test-all:
	@echo "Use ./run_tests.sh for running tests"
	@echo "Examples:"
	@echo "  ./run_tests.sh --all      # All configurations"
	@echo "  ./run_tests.sh --quick    # Quick debug tests" 
	@echo "  ./run_tests.sh --sanitize # Memory safety tests"
	@echo "  ./run_tests.sh --help     # See all options"

# Clean build files
clean:
	rm -rf $(BUILD_DIR)

# Help targets
help:
	@echo "Available targets:"
	@echo "  all               Build test executable (default: debug)"
	@echo "  debug             Build debug version of test executable"
	@echo "  release           Build optimized release version"
	@echo "  sanitize          Build with memory sanitizers"
	@echo "  test              Build test executable"
	@echo "  test-all          Show test running instructions"
	@echo "  clean             Remove build files"
	@echo "  install-gtest     Show Google Test installation instructions"
	@echo ""
	@echo "Test executable:"
	@echo "  $(TEST_EXECUTABLE)    - Single executable with all tests"
	@echo ""
	@echo "Running tests:"
	@echo "  Use ./run_tests.sh for all test operations:"
	@echo "    ./run_tests.sh --all        # All configurations (debug/release/sanitize)"
	@echo "    ./run_tests.sh --quick      # Quick debug tests only"
	@echo "    ./run_tests.sh --sanitize   # Memory safety tests"
	@echo "    ./run_tests.sh --individual # Run each test suite separately"
	@echo "    ./run_tests.sh --valgrind   # Memory leak checking"
	@echo "    ./run_tests.sh --help       # See all options"
	@echo ""
	@echo "Direct executable usage:"
	@echo "    $(TEST_EXECUTABLE)                        # Run all tests"
	@echo "    $(TEST_EXECUTABLE) --gtest_filter='*ReLU*'    # Run specific tests"
	@echo "    $(TEST_EXECUTABLE) --gtest_list_tests         # List all tests"

# Install google test (help target)
install-gtest:
	@echo "Please install Google Test:"
	@echo "macOS: brew install googletest"
	@echo "Ubuntu/Debian: sudo apt-get install libgtest-dev"
	@echo ""
	@echo "After installation, you can build and run tests with:"
	@echo "  make test-individual    # Run each test suite"
	@echo "  make test-all          # Run comprehensive test suite"