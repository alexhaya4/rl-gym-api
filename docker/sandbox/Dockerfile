FROM python:3.12-slim

# Install only gymnasium and numpy - no network access needed
RUN pip install --no-cache-dir gymnasium numpy

# Create non-root sandbox user with no shell
RUN useradd --uid 1001 --no-create-home --shell /bin/false sandbox

# Create working directory
WORKDIR /sandbox

# Copy the runner script
COPY runner.py /sandbox/runner.py

# Switch to sandbox user
USER sandbox

# No CMD - will be called with arguments
