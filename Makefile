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
TEST_EXECUTABLE = $(BUILD_DIR)/tensor_test

# Main targets
.PHONY: all clean debug release sanitize test test-all test-quick

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

# Build test executable
$(TEST_EXECUTABLE): $(OBJECTS) $(TEST_OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(GTEST_LIBS)

# Run tests
test: $(TEST_EXECUTABLE)
	./$(TEST_EXECUTABLE)

# Run comprehensive test suite
test-all:
	./run_tests.sh --all

# Run quick tests only
test-quick:
	./run_tests.sh --quick

# Clean build files
clean:
	rm -rf $(BUILD_DIR)

# Install google test (help target)
install-gtest:
	@echo "Please install Google Test:"
	@echo "macOS: brew install googletest"
	@echo "Ubuntu/Debian: sudo apt-get install libgtest-dev"