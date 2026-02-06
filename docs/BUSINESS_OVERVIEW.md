# QueryMind Business Overview

## What QueryMind Is
QueryMind is a metadata intelligence platform for BigQuery. It helps teams discover trusted data assets, understand relationships between tables, validate data quality, and generate usable SQL faster.

It combines:
- A metadata pipeline that continuously indexes your data catalog.
- An AI-powered API that answers data questions.
- A chat UI for analysts and business users.

## Why It Exists
Most teams lose time on:
- Finding the right table or column.
- Understanding upstream and downstream dependencies.
- Writing SQL from scratch for common reporting questions.
- Detecting quality issues before reporting.

QueryMind addresses these gaps by turning raw metadata into searchable, explainable, and operationally useful intelligence.

## Core Business Capabilities
1. Metadata discovery
Find the right tables and columns using semantic and keyword search.

2. Lineage visibility
See how data moves across bronze, silver, and gold layers and identify dependency chains.

3. SQL acceleration
Generate SQL for business questions using real schema context and foreign-key relationships.

4. Data quality checks
Validate tables with rule-based checks and receive readable results (including CLI-style report output).

5. Change impact analysis
Run ripple analysis before schema changes to estimate blast radius and risk.

## Who Uses It
- Analysts: Faster table discovery and SQL generation.
- Data engineers: Lineage, impact analysis, and validation checks.
- Data governance teams: Better visibility into metadata quality and PII-related context.
- Product and business stakeholders: Faster access to trusted data explanations.

## What Value It Delivers
- Reduces time-to-answer for analytics questions.
- Lowers reporting risk by exposing quality and dependency issues earlier.
- Improves confidence in SQL by grounding output in metadata and FK graph context.
- Scales institutional knowledge of your warehouse without manual documentation effort.

## High-Level Flow
1. QueryMind extracts table, column, and lineage metadata from BigQuery.
2. It enriches metadata with AI-generated business context.
3. It creates embeddings for semantic retrieval.
4. Users ask questions in the UI or API.
5. QueryMind returns search results, SQL, validation reports, or impact analysis.

## Security and Access Model (Business View)
- Uses Google Cloud runtime identity (service account) for BigQuery and Vertex AI access.
- Designed to run on Cloud Run with server-side API proxying from UI.
- Supports deployment patterns that avoid client-side credential exposure.

## Current Scope
- Built for BigQuery datasets in medallion-style layers.
- Includes API and UI for interactive use.
- Supports production deployment on Cloud Run.

## Expected Outcome
QueryMind becomes a practical "data navigation and trust layer" for your warehouse, helping teams move from question to reliable answer with less manual back-and-forth.
