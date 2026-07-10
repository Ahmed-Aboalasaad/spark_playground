# NYC Taxi Analytics with Apache Spark — Streamlit + PySpark container.
#
# Spark 4.x requires Java 17+, so the image bundles a Java 17 JRE alongside
# Python (pinned rather than taking whatever the Debian base ships — Java 21
# runs Spark too, but trips extra JDK-module-security warnings against
# Spark's Hadoop shim). The app's own Home page downloads NYC TLC parquet
# data at runtime (there is no dataset baked into the image), so a fresh
# container is immediately runnable — just open it and use the Data control
# panel.

FROM eclipse-temurin:17-jre AS jdk

FROM python:3.12-slim

# --- Java 17, copied in from the Temurin stage above. ---------------------- #
COPY --from=jdk /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# --- Remaining system dependencies: libgomp1 (XGBoost's OpenMP runtime),
# curl (healthcheck). -------------------------------------------------------- #
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies (cached separately from app code) --------------- #
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application code ------------------------------------------------------ #
COPY . .

# Writable dirs for downloaded parquet data and trained models. Not part of
# the image's committed content (see .dockerignore) — mount volumes here to
# persist across container restarts:
#   docker run -v taxi-data:/app/data -v taxi-models:/app/models ...
RUN mkdir -p /app/data /app/models
VOLUME ["/app/data", "/app/models"]

ENV STREAMLIT_SERVER_HEADLESS=true \
    PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/Home.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
