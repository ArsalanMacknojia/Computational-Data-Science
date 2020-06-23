# CMPT 353 - Exercise 6
# Arsalan Macknojia

import sys
import pandas as pd
from scipy import stats

OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value: {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value: {more_searches_p:.3g}\n'
    '"Did more/less instructors use the search feature?" p-value: {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value: {more_instr_searches_p:.3g}'
)


def get_new_search_data(df):
    return df.loc[df['uid'] % 2 == 1]


def get_old_search_data(df):
    return df.loc[df['uid'] % 2 == 0]


def search_used_count(df):
    return len(df.loc[df['search_count'] > 0].index)


def search_not_used_count(df):
    return len(df.loc[df['search_count'] < 1].index)


def chi_squared_test(new_search, old_search):
    new_search_used = search_used_count(new_search)
    new_search_not_used = search_not_used_count(new_search)

    old_search_used = search_used_count(old_search)
    old_search_not_used = search_not_used_count(old_search)

    contingency = pd.DataFrame(
        [[new_search_used, new_search_not_used], [old_search_used, old_search_not_used]],
        columns=['search-used', 'search-not-used'], index=pd.Index(['new search count', 'old search count']))

    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return p


def mann_whitney_test(new_search, old_search):
    return stats.mannwhitneyu(new_search['search_count'], old_search['search_count']).pvalue


def main():
    search_file = sys.argv[1]
    search_data = pd.read_json(search_file, orient='records', lines=True)

    # All user data analysis
    new_search = get_new_search_data(search_data)
    old_search = get_old_search_data(search_data)

    more_users_p = chi_squared_test(new_search, old_search)
    more_searches_p = mann_whitney_test(new_search, old_search)

    # Instructor data analysis
    instructor_data = search_data.loc[search_data['is_instructor'] == True]
    new_instr_search = get_new_search_data(instructor_data)
    old_instr_search = get_old_search_data(instructor_data)

    more_instr_p = chi_squared_test(new_instr_search, old_instr_search)
    more_instr_searches_p = mann_whitney_test(new_instr_search, old_instr_search)

    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p=more_users_p,
        more_searches_p=more_searches_p,
        more_instr_p=more_instr_p,
        more_instr_searches_p=more_instr_searches_p,
    ))


if __name__ == '__main__':
    main()
