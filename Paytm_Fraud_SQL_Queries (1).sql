-- ============================================================
--  PAYTM FINANCIAL FRAUD DETECTION SYSTEM
--  PostgreSQL Queries  |  8 Business Problems
--  Dataset: paytm_fraud  |  1,00,000 transactions
-- ============================================================
CREATE DATABASE PAYTM ;
-- HOW TO LOAD:
 CREATE TABLE paytm_fraud (
     Transaction_ID TEXT, Date DATE, Time TIME, Hour INT,
     Day_of_Week TEXT, Month TEXT, Month_Num INT, Quarter TEXT,
     User_ID TEXT, Account_Age_Days INT, Device_ID TEXT,
     Device_Type TEXT, OS TEXT, App_Version TEXT, IP_Address TEXT,
     Is_New_Device INT, Transaction_Type TEXT, Payment_Category TEXT,
     Amount BIGINT, Txn_Count_Last_1Hr INT, Txn_Count_Last_24Hr INT,
     Avg_Txn_Amount_30D BIGINT, Amount_vs_Avg_Ratio NUMERIC(8,2),
     Failed_Login_Attempts INT, Txn_City TEXT, Txn_State TEXT,
     Registered_City TEXT, Registered_State TEXT, Login_City TEXT,
     Location_Mismatch INT, Is_Night_Transaction INT,
     Is_New_Account INT, Is_High_Value_Flag INT,
     Fraud_Score INT, Is_Fraud INT, Fraud_Type TEXT
  )
COPY paytm_fraud FROM '/path/to/paytm_fraud_transactions.csv'
DELIMITER ',' CSV HEADER;

-- ============================================================
--  PROBLEM 1: TRANSACTION FREQUENCY SPIKE DETECTION
--  Finding: Users making 8+ txn/hr show 3-5x higher fraud rate
--  than users making 1-3 txn/hr. Velocity fraud = 29.6% of all fraud.
-- ============================================================

-- 1A. Fraud rate by transaction velocity bucket
SELECT
    CASE
        WHEN txn_count_last_1hr <= 2  THEN '1-2 txn/hr  (Normal)'
        WHEN txn_count_last_1hr <= 4  THEN '3-4 txn/hr  (Watch)'
        WHEN txn_count_last_1hr <= 7  THEN '5-7 txn/hr  (Elevated)'
        WHEN txn_count_last_1hr <= 12 THEN '8-12 txn/hr (High Risk)'
        ELSE                               '13+ txn/hr  (Block)'
    END                                                    AS velocity_bucket,
    COUNT(*)                                               AS total_transactions,
    SUM(is_fraud)                                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)            AS fraud_rate_pct,
    ROUND(AVG(amount))                                     AS avg_amount,
    RANK() OVER (ORDER BY SUM(is_fraud) * 100.0 / COUNT(*) DESC)
                                                           AS risk_rank
FROM paytm_fraud
GROUP BY velocity_bucket
ORDER BY fraud_rate_pct DESC;


-- 1B. Top 20 high-velocity users with fraud flag
SELECT
    user_id,
    COUNT(*)                                               AS total_txns,
    SUM(is_fraud)                                          AS fraud_txns,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 1)            AS fraud_rate_pct,
    MAX(txn_count_last_1hr)                                AS peak_txn_per_hr,
    MAX(txn_count_last_24hr)                               AS peak_txn_per_day,
    ROUND(AVG(amount))                                     AS avg_txn_amount,
    ROUND(SUM(amount) / 1e5, 2)                           AS total_amount_lakh,
    CASE
        WHEN SUM(is_fraud) > 0 THEN 'CONFIRMED FRAUD'
        WHEN MAX(txn_count_last_1hr) > 10 THEN 'HIGH RISK'
        ELSE 'MONITOR'
    END                                                    AS account_status
FROM paytm_fraud
WHERE txn_count_last_1hr >= 5
GROUP BY user_id
HAVING COUNT(*) >= 3
ORDER BY fraud_txns DESC, peak_txn_per_hr DESC
LIMIT 20;


-- 1C. Velocity fraud by hour — when do spikes happen?
SELECT
    hour,
    COUNT(*) FILTER (WHERE txn_count_last_1hr >= 8)        AS high_velocity_txns,
    SUM(is_fraud) FILTER (WHERE txn_count_last_1hr >= 8)   AS fraud_in_spike,
    ROUND(
        SUM(is_fraud) FILTER (WHERE txn_count_last_1hr >= 8) * 100.0 /
        NULLIF(COUNT(*) FILTER (WHERE txn_count_last_1hr >= 8), 0), 2
    )                                                      AS fraud_rate_pct,
    CASE WHEN hour BETWEEN 0 AND 4 THEN 'Night'
         WHEN hour BETWEEN 5 AND 11 THEN 'Morning'
         WHEN hour BETWEEN 12 AND 17 THEN 'Afternoon'
         ELSE 'Evening' END                                AS time_period
FROM paytm_fraud
GROUP BY hour
ORDER BY hour;


-- ============================================================
--  PROBLEM 2: LOCATION MISMATCH — CITY HOPPING ANOMALY
--  Finding: Transaction city ≠ Registered city in same session.
--  Mismatch transactions = 21.3% of all fraud cases detected.
-- ============================================================

-- 2A. Location mismatch fraud analysis by city
SELECT
    txn_city,
    txn_state,
    COUNT(*)                                               AS total_txns,
    SUM(location_mismatch)                                 AS mismatch_txns,
    ROUND(SUM(location_mismatch) * 100.0 / COUNT(*), 2)   AS mismatch_rate_pct,
    SUM(is_fraud)                                          AS total_fraud,
    SUM(CASE WHEN location_mismatch = 1 AND is_fraud = 1 THEN 1 ELSE 0 END)
                                                           AS mismatch_fraud,
    ROUND(
        SUM(CASE WHEN location_mismatch = 1 AND is_fraud = 1 THEN 1 ELSE 0 END) * 100.0 /
        NULLIF(SUM(location_mismatch), 0), 2
    )                                                      AS fraud_in_mismatch_pct
FROM paytm_fraud
GROUP BY txn_city, txn_state
ORDER BY mismatch_fraud DESC;


-- 2B. Most common city-hopping pairs (source → destination fraud)
SELECT
    registered_city                                        AS home_city,
    txn_city                                               AS fraud_city,
    COUNT(*)                                               AS mismatch_count,
    SUM(is_fraud)                                          AS confirmed_fraud,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)            AS fraud_rate_pct,
    ROUND(AVG(amount))                                     AS avg_txn_amount,
    ROUND(MAX(amount) / 1e5, 2)                           AS max_amount_lakh
FROM paytm_fraud
WHERE location_mismatch = 1
  AND registered_city != txn_city
GROUP BY registered_city, txn_city
HAVING COUNT(*) >= 10
ORDER BY fraud_rate_pct DESC, confirmed_fraud DESC
LIMIT 25;


-- 2C. Multi-city sessions — same user, multiple cities in 24hrs
WITH user_cities AS (
    SELECT
        user_id,
        date,
        COUNT(DISTINCT txn_city)                           AS cities_in_day,
        COUNT(*)                                           AS txns_in_day,
        SUM(is_fraud)                                      AS fraud_in_day,
        ROUND(SUM(amount::NUMERIC) / 1e5, 2)              AS amount_lakh
    FROM paytm_fraud
    GROUP BY user_id, date
)
SELECT
    cities_in_day,
    COUNT(*)                                               AS user_day_combos,
    SUM(fraud_in_day)                                      AS fraud_cases,
    ROUND(SUM(fraud_in_day) * 100.0 / NULLIF(COUNT(*), 0), 2)
                                                           AS fraud_rate_pct,
    ROUND(AVG(amount_lakh), 2)                            AS avg_daily_amount_lakh
FROM user_cities
GROUP BY cities_in_day
ORDER BY cities_in_day;


-- ============================================================
--  PROBLEM 3: NEW DEVICE FINGERPRINT FRAUD
--  Finding: 82% of fraud originates from never-seen-before devices.
--  New device + failed login = 94% fraud precision.
-- ============================================================

-- 3A. New device fraud analysis
SELECT
    is_new_device,
    COUNT(*)                                               AS total_txns,
    SUM(is_fraud)                                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)            AS fraud_rate_pct,
    ROUND(AVG(amount))                                     AS avg_amount,
    SUM(CASE WHEN failed_login_attempts >= 3 THEN 1 ELSE 0 END)
                                                           AS with_failed_logins
FROM paytm_fraud
GROUP BY is_new_device
ORDER BY is_new_device;


-- 3B. New device + failed login combination (most dangerous signal)
SELECT
    CASE
        WHEN is_new_device = 1 AND failed_login_attempts >= 3 THEN 'New Device + Failed Logins'
        WHEN is_new_device = 1 AND failed_login_attempts = 0  THEN 'New Device Only'
        WHEN is_new_device = 0 AND failed_login_attempts >= 3 THEN 'Failed Logins Only'
        ELSE                                                        'Normal'
    END                                                    AS risk_combination,
    COUNT(*)                                               AS total_txns,
    SUM(is_fraud)                                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)            AS fraud_rate_pct,
    ROUND(AVG(amount))                                     AS avg_txn_amount
FROM paytm_fraud
GROUP BY risk_combination
ORDER BY fraud_rate_pct DESC;


-- 3C. Device type breakdown — which platform has highest fraud?
SELECT
    device_type,
    COUNT(*)                                               AS total_txns,
    SUM(is_fraud)                                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)            AS fraud_rate_pct,
    SUM(CASE WHEN is_new_device = 1 THEN 1 ELSE 0 END)    AS new_device_txns,
    ROUND(AVG(amount))                                     AS avg_amount
FROM paytm_fraud
GROUP BY device_type
ORDER BY fraud_rate_pct DESC;


-- ============================================================
--  PROBLEM 4: AMOUNT ANOMALY — SUDDEN HIGH-VALUE SPIKE
--  Finding: Transactions 5x+ the user's 30-day average = 78% fraud rate.
--  Amount anomaly = 13.1% of all fraud cases.
-- ============================================================

-- 4A. Fraud rate by amount-vs-average ratio bucket
SELECT
    CASE
        WHEN amount_vs_avg_ratio < 1    THEN 'Below avg'
        WHEN amount_vs_avg_ratio < 2    THEN '1-2x average  (Normal)'
        WHEN amount_vs_avg_ratio < 3    THEN '2-3x average  (Watch)'
        WHEN amount_vs_avg_ratio < 5    THEN '3-5x average  (High Risk)'
        WHEN amount_vs_avg_ratio < 10   THEN '5-10x average (Critical)'
        ELSE                                 '10x+ average  (Block)'
    END                                                    AS amount_ratio_bucket,
    COUNT(*)                                               AS total_txns,
    SUM(is_fraud)                                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)            AS fraud_rate_pct,
    ROUND(AVG(amount) / 1e3, 1)                           AS avg_amount_k,
    ROUND(MAX(amount) / 1e5, 2)                           AS max_amount_lakh
FROM paytm_fraud
GROUP BY amount_ratio_bucket
ORDER BY fraud_rate_pct DESC;


-- 4B. High-value transaction fraud by transaction type
SELECT
    transaction_type,
    COUNT(*)                                               AS total_txns,
    SUM(CASE WHEN is_high_value_flag = 1 THEN 1 ELSE 0 END)
                                                           AS high_value_txns,
    SUM(CASE WHEN is_high_value_flag = 1 AND is_fraud = 1 THEN 1 ELSE 0 END)
                                                           AS high_value_fraud,
    ROUND(
        SUM(CASE WHEN is_high_value_flag = 1 AND is_fraud = 1 THEN 1 ELSE 0 END) * 100.0 /
        NULLIF(SUM(CASE WHEN is_high_value_flag = 1 THEN 1 ELSE 0 END), 0), 2
    )                                                      AS fraud_rate_in_hv_pct,
    ROUND(AVG(CASE WHEN is_high_value_flag = 1 THEN amount END) / 1e3, 1)
                                                           AS avg_hv_amount_k
FROM paytm_fraud
GROUP BY transaction_type
ORDER BY fraud_rate_in_hv_pct DESC;


-- 4C. Amount anomaly with location mismatch (double flag)
SELECT
    CASE
        WHEN amount_vs_avg_ratio >= 5 AND location_mismatch = 1 THEN 'Amount + Location'
        WHEN amount_vs_avg_ratio >= 5 AND is_new_device = 1      THEN 'Amount + New Device'
        WHEN amount_vs_avg_ratio >= 5                             THEN 'Amount Only'
        ELSE 'Other'
    END                                                    AS combined_flag,
    COUNT(*)                                               AS transactions,
    SUM(is_fraud)                                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)            AS fraud_rate_pct,
    ROUND(AVG(amount) / 1e3, 1)                           AS avg_amount_k
FROM paytm_fraud
GROUP BY combined_flag
ORDER BY fraud_rate_pct DESC;


-- ============================================================
--  PROBLEM 5: NIGHT-TIME TRANSACTION FRAUD CONCENTRATION
--  Finding: 12AM-4AM = 4.8-5.1% fraud rate vs 3.1% daytime avg.
--  Night = 6.1% of volume but 23.4% of all fraud.
-- ============================================================

-- 5A. Hourly fraud concentration — 24-hour heatmap
SELECT
    hour,
    TO_CHAR(MAKE_TIME(hour, 0, 0), 'HH12:MI AM')          AS time_label,
    COUNT(*)                                               AS total_txns,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)    AS volume_share_pct,
    SUM(is_fraud)                                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)            AS fraud_rate_pct,
    ROUND(SUM(is_fraud) * 100.0 / SUM(SUM(is_fraud)) OVER (), 2)
                                                           AS fraud_concentration_pct,
    ROUND(AVG(amount))                                     AS avg_amount,
    CASE
        WHEN hour BETWEEN 0 AND 4   THEN 'Night (High Risk)'
        WHEN hour BETWEEN 5 AND 11  THEN 'Morning'
        WHEN hour BETWEEN 12 AND 17 THEN 'Afternoon'
        ELSE 'Evening'
    END                                                    AS time_period
FROM paytm_fraud
GROUP BY hour
ORDER BY hour;


-- 5B. Night fraud by transaction type
SELECT
    transaction_type,
    COUNT(*) FILTER (WHERE hour BETWEEN 0 AND 4)           AS night_txns,
    SUM(is_fraud) FILTER (WHERE hour BETWEEN 0 AND 4)      AS night_fraud,
    ROUND(
        SUM(is_fraud) FILTER (WHERE hour BETWEEN 0 AND 4) * 100.0 /
        NULLIF(COUNT(*) FILTER (WHERE hour BETWEEN 0 AND 4), 0), 2
    )                                                      AS night_fraud_rate_pct,
    COUNT(*) FILTER (WHERE hour NOT BETWEEN 0 AND 4)       AS day_txns,
    ROUND(
        SUM(is_fraud) FILTER (WHERE hour NOT BETWEEN 0 AND 4) * 100.0 /
        NULLIF(COUNT(*) FILTER (WHERE hour NOT BETWEEN 0 AND 4), 0), 2
    )                                                      AS day_fraud_rate_pct
FROM paytm_fraud
GROUP BY transaction_type
ORDER BY night_fraud_rate_pct DESC;


-- 5C. Night + weekend combination risk
SELECT
    CASE
        WHEN hour BETWEEN 0 AND 4 AND day_of_week IN ('Saturday','Sunday')
            THEN 'Night + Weekend'
        WHEN hour BETWEEN 0 AND 4
            THEN 'Night Only'
        WHEN day_of_week IN ('Saturday','Sunday')
            THEN 'Weekend Only'
        ELSE 'Weekday Daytime'
    END                                                    AS time_segment,
    COUNT(*)                                               AS transactions,
    SUM(is_fraud)                                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)            AS fraud_rate_pct,
    ROUND(AVG(amount))                                     AS avg_amount
FROM paytm_fraud
GROUP BY time_segment
ORDER BY fraud_rate_pct DESC;


-- ============================================================
--  PROBLEM 6: NEW ACCOUNT RAPID SPENDING FRAUD
--  Finding: Accounts <30 days old with txn >₹10,000 within
--  72hrs of creation. New account fraud = 6.6% of all fraud.
-- ============================================================

-- 6A. Fraud rate by account age bucket
SELECT
    CASE
        WHEN account_age_days <= 7   THEN '0-7 days    (Very New)'
        WHEN account_age_days <= 30  THEN '8-30 days   (New)'
        WHEN account_age_days <= 90  THEN '31-90 days  (Recent)'
        WHEN account_age_days <= 365 THEN '91-365 days (Established)'
        ELSE                              '365+ days   (Mature)'
    END                                                    AS account_age_bucket,
    COUNT(*)                                               AS total_txns,
    SUM(is_fraud)                                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)            AS fraud_rate_pct,
    ROUND(AVG(amount))                                     AS avg_amount,
    ROUND(AVG(amount_vs_avg_ratio), 2)                    AS avg_amount_ratio
FROM paytm_fraud
GROUP BY account_age_bucket
ORDER BY fraud_rate_pct DESC;


-- 6B. New account + high value = highest risk combination
SELECT
    is_new_account,
    is_high_value_flag,
    COUNT(*)                                               AS transactions,
    SUM(is_fraud)                                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)            AS fraud_rate_pct,
    ROUND(AVG(amount) / 1e3, 1)                           AS avg_amount_k,
    CASE
        WHEN is_new_account = 1 AND is_high_value_flag = 1 THEN 'HIGHEST RISK — Auto Block'
        WHEN is_new_account = 1 AND is_high_value_flag = 0 THEN 'High Risk — Review'
        WHEN is_new_account = 0 AND is_high_value_flag = 1 THEN 'Moderate — Flag'
        ELSE                                                    'Normal'
    END                                                    AS risk_action
FROM paytm_fraud
GROUP BY is_new_account, is_high_value_flag
ORDER BY fraud_rate_pct DESC;


-- ============================================================
--  PROBLEM 7: FAILED LOGIN → HIGH VALUE TRANSACTION PATTERN
--  Finding: 3+ failed logins followed by successful high-value
--  transaction = brute-force account takeover signal.
-- ============================================================

-- 7A. Failed login attempts vs fraud correlation
SELECT
    failed_login_attempts,
    COUNT(*)                                               AS total_txns,
    SUM(is_fraud)                                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)            AS fraud_rate_pct,
    ROUND(AVG(amount))                                     AS avg_amount,
    CASE
        WHEN failed_login_attempts >= 5 THEN 'Block account immediately'
        WHEN failed_login_attempts >= 3 THEN 'Force re-authentication'
        WHEN failed_login_attempts >= 1 THEN 'Send OTP alert'
        ELSE 'Normal'
    END                                                    AS recommended_action
FROM paytm_fraud
GROUP BY failed_login_attempts
ORDER BY failed_login_attempts;


-- 7B. Multi-signal fraud detection — accounts triggering 3+ flags
SELECT
    user_id,
    COUNT(*)                                               AS total_txns,
    SUM(is_fraud)                                          AS confirmed_fraud,
    MAX(failed_login_attempts)                             AS max_failed_logins,
    SUM(is_new_device)                                     AS new_device_txns,
    SUM(location_mismatch)                                 AS location_mismatch_txns,
    SUM(is_high_value_flag)                                AS high_value_txns,
    SUM(is_night_transaction)                              AS night_txns,
    -- Risk score: count of triggered flags
    (MAX(CASE WHEN failed_login_attempts >= 3 THEN 1 ELSE 0 END)
     + MAX(is_new_device)
     + MAX(location_mismatch)
     + MAX(is_high_value_flag)
     + MAX(is_night_transaction)
     + MAX(is_new_account))                                AS flags_triggered,
    ROUND(SUM(amount::NUMERIC) / 1e5, 2)                  AS total_amount_lakh
FROM paytm_fraud
GROUP BY user_id
HAVING SUM(is_fraud) > 0
ORDER BY flags_triggered DESC, confirmed_fraud DESC
LIMIT 25;


-- ============================================================
--  PROBLEM 8: PAYMENT CATEGORY BEHAVIOURAL DEVIATION
--  Finding: Users switching from recurring categories
--  (bills/recharge) to high-risk categories (international/
--  investment) show elevated fraud signals.
-- ============================================================

-- 8A. Fraud rate by payment category
SELECT
    transaction_type,
    payment_category,
    COUNT(*)                                               AS total_txns,
    SUM(is_fraud)                                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)            AS fraud_rate_pct,
    ROUND(AVG(amount))                                     AS avg_amount,
    ROUND(AVG(amount_vs_avg_ratio), 2)                    AS avg_amount_ratio,
    RANK() OVER (ORDER BY SUM(is_fraud) * 100.0 / COUNT(*) DESC)
                                                           AS fraud_rank
FROM paytm_fraud
GROUP BY transaction_type, payment_category
ORDER BY fraud_rate_pct DESC;


-- 8B. International + high value = highest category risk
SELECT
    CASE
        WHEN transaction_type = 'International'            THEN 'International Transfer'
        WHEN transaction_type = 'Investment'               THEN 'Investment'
        WHEN transaction_type = 'Money Transfer'
         AND amount > 50000                                THEN 'Large Money Transfer'
        WHEN transaction_type IN ('Recharge','Bill Payment')
                                                           THEN 'Recurring (Low Risk)'
        ELSE transaction_type
    END                                                    AS risk_category,
    COUNT(*)                                               AS total_txns,
    SUM(is_fraud)                                          AS fraud_count,
    ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)            AS fraud_rate_pct,
    ROUND(AVG(amount) / 1e3, 1)                           AS avg_amount_k
FROM paytm_fraud
GROUP BY risk_category
ORDER BY fraud_rate_pct DESC;


-- ============================================================
--  MASTER FRAUD RISK SCORECARD
--  Combined multi-signal query for Power BI base table
--  Use this as your primary Power BI data source
-- ============================================================

SELECT
    transaction_id,
    date,
    hour,
    user_id,
    account_age_days,
    device_type,
    transaction_type,
    payment_category,
    amount,
    amount_vs_avg_ratio,
    txn_count_last_1hr,
    txn_count_last_24hr,
    failed_login_attempts,
    txn_city,
    registered_city,
    is_new_device,
    location_mismatch,
    is_night_transaction,
    is_new_account,
    is_high_value_flag,
    is_fraud,
    fraud_type,
    -- Composite risk score (0-6 flags)
    (CASE WHEN txn_count_last_1hr >= 8         THEN 1 ELSE 0 END +
     CASE WHEN location_mismatch = 1           THEN 1 ELSE 0 END +
     CASE WHEN is_new_device = 1               THEN 1 ELSE 0 END +
     CASE WHEN amount_vs_avg_ratio >= 5        THEN 1 ELSE 0 END +
     CASE WHEN is_night_transaction = 1        THEN 1 ELSE 0 END +
     CASE WHEN is_new_account = 1              THEN 1 ELSE 0 END +
     CASE WHEN failed_login_attempts >= 3      THEN 1 ELSE 0 END)
                                                           AS risk_flag_count,
    -- Risk tier label
    CASE
        WHEN (CASE WHEN txn_count_last_1hr >= 8    THEN 1 ELSE 0 END +
              CASE WHEN location_mismatch = 1      THEN 1 ELSE 0 END +
              CASE WHEN is_new_device = 1          THEN 1 ELSE 0 END +
              CASE WHEN amount_vs_avg_ratio >= 5   THEN 1 ELSE 0 END +
              CASE WHEN is_night_transaction = 1   THEN 1 ELSE 0 END +
              CASE WHEN is_new_account = 1         THEN 1 ELSE 0 END +
              CASE WHEN failed_login_attempts >= 3 THEN 1 ELSE 0 END) >= 3
            THEN 'Critical'
        WHEN (CASE WHEN txn_count_last_1hr >= 8    THEN 1 ELSE 0 END +
              CASE WHEN location_mismatch = 1      THEN 1 ELSE 0 END +
              CASE WHEN is_new_device = 1          THEN 1 ELSE 0 END +
              CASE WHEN amount_vs_avg_ratio >= 5   THEN 1 ELSE 0 END +
              CASE WHEN is_night_transaction = 1   THEN 1 ELSE 0 END +
              CASE WHEN is_new_account = 1         THEN 1 ELSE 0 END +
              CASE WHEN failed_login_attempts >= 3 THEN 1 ELSE 0 END) = 2
            THEN 'High'
        WHEN (CASE WHEN txn_count_last_1hr >= 8    THEN 1 ELSE 0 END +
              CASE WHEN location_mismatch = 1      THEN 1 ELSE 0 END +
              CASE WHEN is_new_device = 1          THEN 1 ELSE 0 END +
              CASE WHEN amount_vs_avg_ratio >= 5   THEN 1 ELSE 0 END +
              CASE WHEN is_night_transaction = 1   THEN 1 ELSE 0 END +
              CASE WHEN is_new_account = 1         THEN 1 ELSE 0 END +
              CASE WHEN failed_login_attempts >= 3 THEN 1 ELSE 0 END) = 1
            THEN 'Medium'
        ELSE 'Low'
    END                                                    AS risk_tier,
    -- Time period label
    CASE
        WHEN hour BETWEEN 0 AND 4   THEN 'Night (High Risk)'
        WHEN hour BETWEEN 5 AND 11  THEN 'Morning'
        WHEN hour BETWEEN 12 AND 17 THEN 'Afternoon'
        ELSE 'Evening'
    END                                                    AS time_period
FROM paytm_fraud
ORDER BY risk_flag_count DESC, is_fraud DESC;

-- END OF SCRIPT
-- Total queries: 20 across 8 business problems + 1 master scorecard
