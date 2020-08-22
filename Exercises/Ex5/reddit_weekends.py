
import sys
import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_TEMPLATE = (
    "Initial (invalid) T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann–Whitney U-test p-value: {utest_p:.3g}"
)


def filter_data(df):
    # Filter data for subreddit Canada for years 2012 and 2013.
    df = df[df['subreddit'] == 'canada']
    df = df.loc[(df['date'] >= '2012-01-01') & (df['date'] <= '2013-12-31')]
    # Generate separate DF for weekdays and weekends.
    df['day'] = df['date'].dt.dayofweek
    weekday = df[df['day'] <= 4]
    weekend = df[df['day'] > 4]
    return weekday, weekend


def get_year_week_number(date):
    year_week = date.isocalendar()
    if year_week[0] in [2012, 2013]:
        return year_week[0], year_week[1]


def main():
    reddit_counts = sys.argv[1]
    counts = pd.read_json(reddit_counts, lines=True)
    weekday, weekend = filter_data(counts)

    # T-Test
    initial_ttest_p = stats.ttest_ind(weekday['comment_count'], weekend['comment_count']).pvalue
    initial_weekday_normality_p = stats.normaltest(weekday['comment_count']).pvalue
    initial_weekend_normality_p = stats.normaltest(weekend['comment_count']).pvalue
    initial_levene_p = stats.levene(weekday['comment_count'], weekend['comment_count']).pvalue

    # Fix 1: transforming data
    transformed_weekday = np.sqrt(weekday['comment_count'])
    transformed_weekend = np.sqrt(weekend['comment_count'])
    transformed_weekday_normality_p = stats.normaltest(transformed_weekday).pvalue
    transformed_weekend_normality_p = stats.normaltest(transformed_weekend).pvalue
    transformed_levene_p = stats.levene(transformed_weekday, transformed_weekend).pvalue

    # Fix 2: Central Limit Theorem
    weekday['year_week'] = weekday['date'].apply(get_year_week_number)
    weekend['year_week'] = weekend['date'].apply(get_year_week_number)
    updated_weekday = weekday.groupby('year_week').aggregate('mean').reset_index()
    updated_weekend = weekend.groupby('year_week').aggregate('mean').reset_index()

    weekly_weekday_normality_p = stats.normaltest(updated_weekday['comment_count']).pvalue
    weekly_weekend_normality_p = stats.normaltest(updated_weekend['comment_count']).pvalue
    weekly_levene_p = stats.levene(updated_weekday['comment_count'], updated_weekend['comment_count']).pvalue
    weekly_ttest_p = stats.ttest_ind(updated_weekday['comment_count'], updated_weekend['comment_count']).pvalue

    # Fix 3: Non-parametric test
    mann_whitney_u_test = stats.mannwhitneyu(weekday['comment_count'], weekend['comment_count']).pvalue

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=initial_ttest_p,
        initial_weekday_normality_p=initial_weekday_normality_p,
        initial_weekend_normality_p=initial_weekend_normality_p,
        initial_levene_p=initial_levene_p,
        transformed_weekday_normality_p=transformed_weekday_normality_p,
        transformed_weekend_normality_p=transformed_weekend_normality_p,
        transformed_levene_p=transformed_levene_p,
        weekly_weekday_normality_p=weekly_weekday_normality_p,
        weekly_weekend_normality_p=weekly_weekend_normality_p,
        weekly_levene_p=weekly_levene_p,
        weekly_ttest_p=weekly_ttest_p,
        utest_p=mann_whitney_u_test,
    ))


if __name__ == '__main__':
    main()
