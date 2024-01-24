
import dynamicrouting_summary as dr
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

def plot_DRephys_behavior():
    beh = dr.get_dfs()['performance']

    def get_passing_sessions(df, dprimethresh):
        sessionlist = []
        passedlist = []
        sessions = df['session_id'].unique().tolist()
        for session in sessions:
            sessiondf = df.loc[df['session_id'] == session]
            dprimeintra = sessiondf['same_modal_dprime'].tolist()
            dprimeinter = sessiondf['cross_modal_dprime'].tolist()
            ## can change passing criteria here
            if (np.sum((np.array(dprimeintra) >= dprimethresh))) >3 and (np.sum((np.array(dprimeinter) >= dprimethresh))) >3:
                passed = 1
            else:
                passed = 0
            sessionlist.append(session)
            passedlist.append(passed)
        summary_df = pd.DataFrame(list(zip(sessionlist, passedlist)), columns = ['session', 'passed'])
        return summary_df

    dr_ephys_filter = ((beh['is_ephys']==True)&
                (beh['is_templeton']==False))
    dr_ephys_beh = beh.loc[dr_ephys_filter]

    beh_summary = get_passing_sessions(dr_ephys_beh, 1.5)

    data = (sum(beh_summary.passed == 0), sum(beh_summary.passed == 1))
    keys = ("fail", "pass")
    palette = 'darksalmon', 'mediumseagreen'

    plt.pie(data, labels = keys, colors = palette, autopct='%.0f%%')
    plt.title('DR ephys sessions passing task-switching criteria')
    plt.show()
    print("n =", len(beh_summary), "sessions")
    print("n =", sum(beh_summary.passed ==1), "passing sessions")

def plot_electrode_yield(structure):
    df = dr.getdfs()['electrodes']
    structure_df = df[df['structure'] == structure]
    sns.countplot(data=structure_df, x='session_id', hue='session_id', legend='full')