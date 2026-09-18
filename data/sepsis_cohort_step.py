"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MIMIC-III Sepsis Cohort Extraction for PINN Training.

sourced from:
https://github.com/matthieukomorowski/AI_Clinician/blob/master/AIClinician_sepsis3_def_160219.m
IDENTIFIES THE COHORT OF PATIENTS WITH SEPSIS in MIMIC-III as used in the AI Clinician (Komorowski, et al [Nature, 2018])
(c) Matthieu Komorowski, Imperial College London 2015-2019

Adapted to Python, and minimally modified, by Jayakumar Subramanian and Taylor Killian.
Further modified for PINN training data extraction.

PURPOSE:
------------------------------
Using the sepsis3 criteria, this script uses the preprocessed intermediate
tables to define a cohort of septic patients, including all observations
24 hours before until 48 hours after presumed onset of sepsis. For each
individually timestamped observation, the following variables are extracted:
  - State variables: SpO2, PaO2, Bilirubin, GCS, Urine output, Lactate
  - Action variables: FiO2, Vasopressor rate, IV fluid volume
  - Trajectory index: icu_id, hours since ICU admission
Rows with insufficient valid variables or trajectories shorter than the
minimum length threshold are removed in a final filtering step, producing
mimic_pinn_v4_filtered.csv.

STEPS:
There are two phases of the following procedure:
  - First to compute the SOFA scores for each patient present in the extracted .csv files
  - Second, recompute the reformatting and filling of missing values with only the presumed septic patients

External files required: Reflabs, Refvitals, sample_and_hold (all saved in the ReferenceFiles folder)

This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the license for details.
"""

import argparse
import pyprind

import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from scipy import stats

from fancyimpute import KNN

parser = argparse.ArgumentParser()
parser.add_argument("--process_raw", action='store_true',
                    help="If specified, additionally save trajectories without normalized features")
parser.add_argument("--save_intermediate", action="store_true",
                    help="If specified, save off intermediate tables used to construct final patient table")
pargs = parser.parse_args()

print('Loading processed files created from database using "preprocess.py"')
abx = pd.read_csv('processed_files/abx.csv', sep='|')
culture = pd.read_csv('processed_files/culture.csv', sep='|')
microbio = pd.read_csv('processed_files/microbio.csv', sep='|')
demog = pd.read_csv('processed_files/demog.csv', sep='|')
ce010 = pd.read_csv('processed_files/ce010000.csv', sep='|')
ce1020 = pd.read_csv('processed_files/ce1000020000.csv', sep='|')
ce2030 = pd.read_csv('processed_files/ce2000030000.csv', sep='|')
ce3040 = pd.read_csv('processed_files/ce3000040000.csv', sep='|')
ce4050 = pd.read_csv('processed_files/ce4000050000.csv', sep='|')
ce5060 = pd.read_csv('processed_files/ce5000060000.csv', sep='|')
ce6070 = pd.read_csv('processed_files/ce6000070000.csv', sep='|')
ce7080 = pd.read_csv('processed_files/ce7000080000.csv', sep='|')
ce8090 = pd.read_csv('processed_files/ce8000090000.csv', sep='|')
ce90100 = pd.read_csv('processed_files/ce90000100000.csv', sep='|')
MV = pd.read_csv('processed_files/mechvent.csv', sep='|')
inputpreadm = pd.read_csv('processed_files/preadm_fluid.csv', sep='|')
inputMV = pd.read_csv('processed_files/fluid_mv.csv', sep='|')
inputCV = pd.read_csv('processed_files/fluid_cv.csv', sep='|')
vasoMV = pd.read_csv('processed_files/vaso_mv.csv', sep='|')
vasoCV = pd.read_csv('processed_files/vaso_cv.csv', sep='|')
UOpreadm = pd.read_csv('processed_files/preadm_uo.csv', sep='|')
UO = pd.read_csv('processed_files/uo.csv', sep='|')
labU = [pd.read_csv('processed_files/labs_ce.csv', sep='|'), pd.read_csv('processed_files/labs_le.csv', sep='|')]

labU[1].rename(columns={'timestp': 'charttime'}, inplace=True)
labU = pd.concat(labU, sort=False, ignore_index=True)

# Initial data manipulations
microbio['charttime'] = microbio['charttime'].fillna(microbio['chartdate'])
del microbio['chartdate']
bacterio = pd.concat([microbio, culture], sort=False, ignore_index=True)

demog['morta_90'].fillna(0, inplace=True)
demog['morta_hosp'].fillna(0, inplace=True)
demog['elixhauser'].fillna(0, inplace=True)

# Keep only the first icustay of an admission (CRITICAL FIX FROM MATLAB CODE)
demog = demog.drop_duplicates(subset=['admittime', 'dischtime'], keep='first')

# Get list of all icustayids since that's what we iterate over through the rest of this script
# The old code had a continuous range of icustayids so it was easy to loop through them with a range(numIDS),
# Since we're only keeping the first icustay of a patient's admission, this is now different...
icustayidlist = list(demog.icustay_id.values)

# Calculate the accurate readmission using the demographics data
# (the SQL code from Komorowski, et al incorrectly cumulatively counts how many icu stays each patient has (preprocess.py:line 414)
# and does a coarse boolean check if this number is >1). A readmission is now correctly defined by
# whether the patient has returned to the ICU within 30 days of being previously discharged.

# This is done by grouping all the discharge times for each patient and using them in a comparison
# with the current row's admission time to see if it's within the 30 day cutoff
subj_dischtime_list = demog.sort_values(by='admittime').groupby('subject_id').apply(lambda df: np.unique(
    df.dischtime.values))  # Create list of discharge times for each patient (output is a dict keyed by 'subject_id')


def determine_readmission(s, dischtimes=subj_dischtime_list, cutoff=3600 * 24 * 30):
    '''
    determine_readmisson evaluates each row of the provided dataframe (designed to operate on the demographics table)
    and chooses whether the current admission occurs within the cutoff of the previous discharge
    (here, cutoff=30 days is the default)
    '''
    subject, admission, discharge = s[['subject_id', 'admittime', 'dischtime']]

    # Check for readmission
    subj_stay_idx = np.where(dischtimes[subject] == discharge)[0][0]
    s['re_admission'] = 0
    if subj_stay_idx > 0:
        if (admission - dischtimes[subject][subj_stay_idx - 1]) <= cutoff:
            s['re_admission'] = 1

    return s


# Apply the above function to determine the appropriate readmissions
demog = demog.apply(determine_readmission, axis=1)


########################################################################
#                    ADDITIONAL HELPER FUNCTIONS
########################################################################

def SAH(input, vitalslab_hold, adjust=0):
    '''Matthieu Komorowski - Imperial College London 2017
    will copy a value in the rows below if the missing values are within the
    hold period for this variable (e.g. 48h for weight, 2h for HR...)
    vitalslab_hold = 2x55 cell (with row1 = strings of names ; row 2 = hold time)'''
    temp = np.copy(input)
    hold = vitalslab_hold.values[0, :]
    nrow, ncol = temp.shape

    lastcharttime = np.zeros(ncol)
    lastvalue = np.zeros(ncol)
    oldstayid = temp[0, 1]

    bar_SAH = pyprind.ProgBar(ncol - (3 + adjust))
    for i in range(3 + adjust, ncol):
        bar_SAH.update()
        for j in range(nrow):
            if oldstayid != temp[j, 1]:
                lastcharttime = np.zeros(ncol)
                lastvalue = np.zeros(ncol)
                oldstayid = temp[j, 1]
            if not np.isnan(temp[j, i]):
                lastcharttime[i] = temp[j, 2]
                lastvalue[i] = temp[j, i]
            if j > 0:
                if (np.isnan(temp[j, i])) and (temp[j, 1] == oldstayid) and (
                        (temp[j, 2] - lastcharttime[i]) <= hold[i - (3 + adjust)] * 3600):
                    temp[j, i] = lastvalue[i]
    return temp


def fixgaps(x):
    '''FIXGAPS Linearly interpolates gaps in a time series
    YOUT=FIXGAPS(YIN) linearly interpolates over NaN
    in the input time series (may be complex), but ignores
    trailing and leading NaN.
    R. Pawlowicz 6/Nov/99'''
    y = np.copy(x)
    bd = np.isnan(x)
    gd = np.arange(len(x))[~bd]
    bd[:min(gd)] = False
    bd[max(gd) + 1:] = False
    y[bd] = interp1d(gd, x[gd])(np.arange(len(x))[bd])
    return y


def deloutabove(a, col_no, a_max):
    a[a[:, col_no] > a_max, col_no] = np.nan
    return a


def deloutbelow(a, col_no, a_min):
    a[a[:, col_no] < a_min, col_no] = np.nan
    return a


# Compute normalized rate of infusion
# if 100 ml of hypertonic fluid (600 mosm/l) is given at 100 ml/h (given in 1h) it is 200 ml of NS equivalent
# so the normalized rate of infusion is 200 ml/h (different volume in same duration)
inputMV['norm_rate_of_infusion'] = inputMV['tev'] * inputMV['rate'] / inputMV['amount']

# Fill-in missing ICUSTAY IDs in bacterio
print('Filling-in missing ICUSTAY IDs in bacterio')
bar = pyprind.ProgBar(len(bacterio.index.tolist()))
# Raw Translation
for i in bacterio.index.tolist():
    bar.update()
    if np.isnan(bacterio.loc[i, 'icustay_id']):
        o = bacterio.loc[i, 'charttime']
        subjectid = bacterio.loc[i, 'subject_id']
        hadmid = bacterio.loc[i, 'hadm_id']
        ii = demog.index[demog['subject_id'] == subjectid].tolist()
        jj = demog.index[(demog['subject_id'] == subjectid) & (demog['hadm_id'] == hadmid)].tolist()
        for j in range(len(ii)):
            if (o >= demog.loc[ii[j], 'intime'] - 48 * 3600) and (o <= demog.loc[ii[j], 'outtime'] + 48 * 3600):
                bacterio.loc[i, 'icustay_id'] = demog.loc[ii[j], 'icustay_id']
            elif len(
                    ii) == 1:  # If we cant confirm from admission and discharge time but there is only 1 admission: it's the one!!
                bacterio.loc[i, 'icustay_id'] = demog.loc[ii[j], 'icustay_id']

print('Filling-in missing ICUSTAY IDs in bacterio - 2')
bar = pyprind.ProgBar(len(bacterio.index.tolist()))
for i in bacterio.index.tolist():
    bar.update()
    if np.isnan(bacterio.loc[i, 'icustay_id']):
        subjectid = bacterio.loc[i, 'subject_id']
        hadmid = bacterio.loc[i, 'hadm_id']
        jj = demog.index[(demog['subject_id'] == subjectid) & (demog['hadm_id'] == hadmid)].tolist()
        if len(jj) == 1:
            bacterio.loc[i, 'icustay_id'] = demog.loc[jj[0], 'icustay_id']

# Fill-in missing ICUSTAY IDs in Antibiotics administration
print('Filling-in missing ICUSTAY IDs in ABx')
bar = pyprind.ProgBar(len(abx.index.tolist()))
for i in abx.index.tolist():
    bar.update()
    if np.isnan(abx.loc[i, 'icustay_id']):
        o = abx.loc[i, 'startdate']  # time of event
        hadmid = abx.loc[i, 'hadm_id']
        ii = demog.index[demog['hadm_id'] == hadmid].tolist()
        for j in range(len(ii)):
            if o >= demog.loc[ii[j], 'intime'] - 48 * 3600 and o <= demog.loc[ii[j], 'outtime'] + 48 * 3600:
                abx.loc[i, 'icustay_id'] = demog.loc[ii[j], 'icustay_id']
            elif len(
                    ii) == 1:  # if we cant confirm from admission and discharge time but there is only 1 admission: it's the one!!
                abx.loc[i, 'icustay_id'] = demog.loc[ii[j], 'icustay_id']

########################################################################
#   Find presumed onset of infection according to sepsis3 guidelines
########################################################################

# METHOD:
# Loop through all administered antibiotics as soon as
# a sample is present within the time window break the loop.

print('Full ICU -- Finding presumed onset of infection according to sepsis3 guidelines')

onset = dict()
num_onset = 0
bar = pyprind.ProgBar(len(icustayidlist))
for icustayid in icustayidlist:
    bar.update()
    onset[icustayid] = np.zeros(3)
    ab = abx.loc[abx['icustay_id'] == icustayid, 'startdate']  # Start time of abx for this icustayid
    bact = bacterio.loc[bacterio['icustay_id'] == icustayid, 'charttime']  # Time of sample
    subj_bact = bacterio.loc[bacterio['icustay_id'] == icustayid, 'subject_id']

    if len(ab) > 0 and len(bact) > 0:  # If we have data for both: proceed
        # Pairwise distances between antibiotic adminstration and requested cultures, in hours
        D = cdist(ab.values.reshape(ab.values.shape[0], 1), bact.values.reshape(bact.values.shape[0], 1)) / 3600
        for i in range(D.shape[0]):  # looping through all rows of adminsitered antibiotics, from early to late
            M, I = np.min(D[i, :]), np.argmin(D[i, :])  # minimum distance in this row
            ab1 = ab.iloc[i]  # timestamp of this value in list of antibiotics
            bact1 = bact.iloc[I]  # timestamp in list of cultures
            if M <= 24 and ab1 <= bact1:  # if ab was first and delay < 24h
                onset[icustayid][0] = subj_bact.iloc[0]
                onset[icustayid][1] = icustayid
                onset[icustayid][2] = ab1  # Onset of infection = abx time
                num_onset += 1
                break
            elif M <= 72 and ab1 >= bact1:  # elseif sample was first and delay < 72h
                onset[icustayid][0] = subj_bact.iloc[0]
                onset[icustayid][1] = icustayid
                onset[icustayid][2] = bact1  # Onset of infection = sample time
                num_onset += 1
                break

# Sum of records found
print('Full ICU -- Number of preliminary, presumed septic trajectories: ', num_onset)

# Replacing item_ids with column numbers from reference tables
print('Full ICU -- Replacing item_ids with column numbers from reference tables')

# Replace itemid in labs with column number
# This will accelerate process later
Reflabs = pd.read_csv("ReferenceFiles/Reflabs.tsv", sep='\t', header=None)
Reflabs_values = np.unique(Reflabs.fillna(-10000))[1:]
Reflabs_id_dict = {}
for r in Reflabs_values:
    try:
        Reflabs_id_dict[r] = np.max(
            np.where(Reflabs.values == r)[0]) + 1  # for row: +1 due to Index correction python
    except:
        print(r)
        break
itemid_col = labU.columns.tolist().index('itemid')
labU_temp = labU.values
for c in range(labU_temp.shape[0]):
    labU_temp[c, itemid_col] = Reflabs_id_dict[labU_temp[c, itemid_col]]
for i, c in enumerate(labU.columns.tolist()):
    labU.loc[:, c] = labU_temp[:, i]

# Replace itemid in vitals with col number
Refvitals = pd.read_csv("ReferenceFiles/Refvitals.tsv", sep='\t', header=None)
Refvitals_values = np.unique(Refvitals.fillna(-10000))[1:]
Refvitals_id_dict = {}
for r in Refvitals_values:
    Refvitals_id_dict[r] = np.max(
        np.where(Refvitals.values == r)[0]) + 1  # +1 due to index correction for Python from MATLAB
ce_dfs = [ce010, ce1020, ce2030, ce3040, ce4050, ce5060, ce6070, ce7080, ce8090, ce90100]
for ce_df in ce_dfs:
    itemid_col = ce_df.columns.tolist().index('itemid')
    ce_df_temp = ce_df.values
    for c in range(ce_df_temp.shape[0]):
        ce_df_temp[c, itemid_col] = Refvitals_id_dict[ce_df_temp[c, itemid_col]]
    for i, c in enumerate(ce_df.columns.tolist()):
        ce_df.loc[:, c] = ce_df_temp[:, i]

# ########################################################################
#           INITIAL REFORMAT WITH CHARTEVENTS, LABS AND MECHVENT
# ########################################################################

print(' Full ICU --  Making an array with all unique charttime (1 per row) and all items in columns.')
reformat = np.nan * np.ones((6000000, 76))  # Final table
qstime = dict()
winb4 = 25  # Lower limit for inclusion of data (24h before time flag)
winaft = 49  # Upper limit (48h after)
irow = 0  # Recording row for summary table
bar = pyprind.ProgBar(len(icustayidlist))
for icustayid in icustayidlist:
    qstime[icustayid] = np.zeros(4)
    bar.update()
    qst = onset[icustayid][2]  # flag for presumed infection
    if qst > 0:  # if we have a flag
        d1 = demog.loc[demog['icustay_id'] == icustayid, ['age', 'dischtime']].values[
            0]  # Age of patient + discharge time
        if d1[0] > 6574:  # If older than 18 years old
            # CHARTEVENTS
            if (icustayid - 200000) < 10000:
                temp = ce010
            elif (icustayid - 200000) < 20000:
                temp = ce1020
            elif (icustayid - 200000) < 30000:
                temp = ce2030
            elif (icustayid - 200000) < 40000:
                temp = ce3040
            elif (icustayid - 200000) < 50000:
                temp = ce4050
            elif (icustayid - 200000) < 60000:
                temp = ce5060
            elif (icustayid - 200000) < 70000:
                temp = ce6070
            elif (icustayid - 200000) < 80000:
                temp = ce7080
            elif (icustayid - 200000) < 90000:
                temp = ce8090
            else:
                temp = ce90100
            temp = temp[temp['icustay_id'] == icustayid]

            ii = (temp['charttime'] >= qst - (winb4 + 4) * 3600) & (
                        temp['charttime'] <= qst + (winaft + 4) * 3600)  # Time period of interest -4h and +4h
            temp = temp.loc[ii]  # Only time period of interest

            # LABEVENTS
            ii = labU['icustay_id'] == icustayid
            temp2 = labU.loc[ii]
            ii = (temp2['charttime'] >= qst - (winb4 + 4) * 3600) & (
                        temp2['charttime'] <= qst + (winaft + 4) * 3600)  # Time period of interest -4h and +4h
            temp2 = temp2.loc[ii]  # Only time period of interest

            # Mech Vent + ?extubated
            ii = MV['icustay_id'] == icustayid
            temp3 = MV.loc[ii]
            ii = (temp3['charttime'] >= qst - (winb4 + 4) * 3600) & (
                        temp3['charttime'] <= qst + (winaft + 4) * 3600)  # Time period of interest -4h and +4h
            temp3 = temp3.loc[ii]  # only time period of interest
            # t = np.unique(pd.concat([temp['charttime'], temp2['charttime'], temp3['charttime']],
            #                         ignore_index=True).values)  # List of unique timestamps from all 3 sources / sorted in ascending order

            # --- Urine Output ---
            ii_uo = UO['icustay_id'] == icustayid
            temp_uo = UO.loc[ii_uo].sort_values('charttime')
            ii_uo_range = (temp_uo['charttime'] >= qst - (winb4 + 4) * 3600) & (
                        temp_uo['charttime'] <= qst + (winaft + 4) * 3600)
            temp_uo = temp_uo.loc[ii_uo_range]
            uo_times = temp_uo['charttime'].values
            uo_values = temp_uo['value'].values

            v_mv = vasoMV[vasoMV['icustay_id'] == icustayid]
            v_cv = vasoCV[vasoCV['icustay_id'] == icustayid]

            # ===============================
            # IV FLUIDS
            # ===============================

            # MV fluids
            f_mv = inputMV[inputMV['icustay_id'] == icustayid]

            startt = f_mv['starttime'].values
            endt = f_mv['endtime'].values
            rate = f_mv['norm_rate_of_infusion'].values
            f_mv = f_mv.values

            # CV fluids (bolus)
            f_cv = inputCV[inputCV['icustay_id'] == icustayid]
            f_cv = f_cv.values

            # pre ICU fluids
            pread = inputpreadm[inputpreadm['icustay_id'] == icustayid]['inputpreadm']

            if len(pread) > 0:
                totvol = np.nansum(pread)
            else:
                totvol = 0

            t_sources = [temp['charttime'], temp2['charttime'], temp3['charttime'], temp_uo['charttime']]
            if not v_mv.empty:
                t_sources.append(v_mv['starttime'])
                t_sources.append(v_mv['endtime'])
            if not v_cv.empty:
                t_sources.append(v_cv['charttime'])

            # fluids timestamps
            if len(startt) > 0:
                t_sources.append(pd.Series(startt))
            if len(endt) > 0:
                t_sources.append(pd.Series(endt))
            if len(inputCV[inputCV['icustay_id'] == icustayid]) > 0:
                t_sources.append(inputCV[inputCV['icustay_id'] == icustayid]['charttime'])

            t = np.unique(pd.concat(t_sources, ignore_index=True).values)

            t = t[(t >= qst - (winb4 + 4) * 3600) & (t <= qst + (winaft + 4) * 3600)]

            if len(t) > 0:
                patient_cumulative_uo = 0

                for i in range(len(t)):
                    t_curr = t[i]

                    t_prev = t[i - 1] if i > 0 else t_curr - 3600

                    # ===============================
                    # IV FLUIDS (ml per timestep)
                    # ===============================

                    t0 = t_prev
                    t1 = t_curr

                    # infusion (MV)
                    infu = np.nansum(
                        rate * (endt - startt) * ((endt <= t1) & (startt >= t0)) / 3600
                        + rate * (endt - t0) * ((startt <= t0) & (endt <= t1) & (endt >= t0)) / 3600
                        + rate * (t1 - startt) * ((startt >= t0) & (endt >= t1) & (startt <= t1)) / 3600
                        + rate * (t1 - t0) * ((endt >= t1) & (startt <= t0)) / 3600
                    )

                    # bolus (MV + CV)
                    bolus = np.nansum(
                        f_mv[(np.isnan(f_mv[:, 5])) & (f_mv[:, 1] >= t0) & (f_mv[:, 1] <= t1), 6]
                    ) + np.nansum(
                        f_cv[(f_cv[:, 1] >= t0) & (f_cv[:, 1] <= t1), 4]
                    )

                    #  timestep fluids
                    step_fluids = np.nansum([infu, bolus])

                    # fluids
                    totvol += step_fluids

                    mask = (uo_times > t_prev) & (uo_times <= t_curr)
                    step_uo = np.nansum(uo_values[mask])
                    patient_cumulative_uo += step_uo

                    current_vaso = 0

                    mv_mask = (t_curr >= v_mv['starttime']) & (t_curr <= v_mv['endtime'])
                    if np.any(mv_mask):

                        current_vaso = v_mv.loc[mv_mask, 'rate_std'].max()

                    elif not v_cv.empty:
                        cv_mask = (v_cv['charttime'] <= t_curr) & (v_cv['charttime'] >= t_curr - 7200)
                        if np.any(cv_mask):
                            current_vaso = v_cv.loc[cv_mask].sort_values('charttime').iloc[-1]['rate_std']

                    reformat[irow, 0] = i + 1
                    reformat[irow, 1] = icustayid
                    reformat[irow, 2] = t_curr

                    reformat[irow, 70] = step_uo
                    reformat[irow, 71] = patient_cumulative_uo

                    # Vaso
                    reformat[irow, 72] = current_vaso
                    # fluids
                    reformat[irow, 73] = step_fluids
                    reformat[irow, 74] = totvol  # total fluids

                    # CHARTEVENTS
                    ii = temp['charttime'] == t_curr
                    col = temp.loc[ii, 'itemid']
                    value = temp.loc[ii, 'valuenum']

                    reformat[irow, 2 + col.astype(int).values] = value.values  # Store available values

                    # LAB VALUES
                    ii = temp2['charttime'] == t_curr
                    col = temp2.loc[ii, 'itemid']
                    value = temp2.loc[ii, 'valuenum']
                    reformat[irow, 30 + col.astype(int).values] = value.values  # Store available values

                    # Mechanical Ventilation
                    ii = temp3['charttime'] == t_curr
                    if np.nansum(ii) > 0:
                        col = temp3.loc[ii, 'mechvent']
                        value = temp3.loc[ii, 'extubated']
                        reformat[irow, 66] = col.values[0]  # Store available values
                        reformat[irow, 67] = value.values[0]  # Store available values
                    else:
                        reformat[irow, 66] = np.nan
                        reformat[irow, 67] = np.nan
                    irow += 1

                qstime[icustayid][
                    0] = qst  # Flag for presumed infection / this is time of sepsis if SOFA >=2 for this patient
                # SAVE FIRST and LAST TIMESTAMPS, in QSTIME, for each ICUSTAYID
                qstime[icustayid][1] = t[0]  # First timestamp
                qstime[icustayid][2] = t[-1]  # Last timestamp
                qstime[icustayid][3] = d1[1]  # Discharge time

reformat = np.delete(reformat, range(irow, len(reformat)), axis=0)  # Delete unused rows

########################################################################
#                                   OUTLIERS
########################################################################
print('Full ICU -- Handling outliers')

# Weight
reformat = deloutabove(reformat, 4, 300)

# Heart Rate
reformat = deloutabove(reformat, 7, 250)

# Blood Pressure
reformat = deloutabove(reformat, 8, 300)
reformat = deloutbelow(reformat, 9, 0)
reformat = deloutabove(reformat, 9, 200)
reformat = deloutbelow(reformat, 10, 0)
reformat = deloutabove(reformat, 10, 200)

# Respiratory Rate
reformat = deloutabove(reformat, 11, 80)

# SpO2
reformat = deloutabove(reformat, 12, 150)
reformat[reformat[:, 12] > 100, 12] = 100

# Temperature
reformat[(reformat[:, 13] > 90) & (np.isnan(reformat[:, 14])), 14] = reformat[
    (reformat[:, 13] > 90) & (np.isnan(reformat[:, 14])), 13]
reformat = deloutabove(reformat, 13, 90)

# Interface / is in col 22
# FiO2
reformat = deloutabove(reformat, 22, 100)
reformat[reformat[:, 22] < 1, 22] = reformat[reformat[:, 22] < 1, 22] * 100

reformat = deloutbelow(reformat, 22, 20)
reformat = deloutabove(reformat, 23, 1.5)

# O2 FLOW
reformat = deloutabove(reformat, 24, 70)

# PEEP
reformat = deloutbelow(reformat, 25, 0)
reformat = deloutabove(reformat, 25, 40)

# Total Volume
reformat = deloutabove(reformat, 26, 1800)

# Mean Volume
reformat = deloutabove(reformat, 27, 50)

# Potassium
reformat = deloutbelow(reformat, 31, 1)
reformat = deloutabove(reformat, 31, 15)

# Sodium
reformat = deloutbelow(reformat, 32, 95)
reformat = deloutabove(reformat, 32, 178)

# Chloride
reformat = deloutbelow(reformat, 33, 70)
reformat = deloutabove(reformat, 33, 150)

# Glucose
reformat = deloutbelow(reformat, 34, 1)
reformat = deloutabove(reformat, 34, 1000)

# Creatinine
reformat = deloutabove(reformat, 36, 150)

# Magnesium
reformat = deloutabove(reformat, 37, 10)

# Calcium
reformat = deloutabove(reformat, 38, 20)

# Ionized Calcium
reformat = deloutabove(reformat, 39, 5)

# CO2
reformat = deloutabove(reformat, 40, 120)

# SGPT/SGOT
reformat = deloutabove(reformat, 41, 10000)
reformat = deloutabove(reformat, 42, 10000)

# Hb/Ht
reformat = deloutabove(reformat, 49, 20)
reformat = deloutabove(reformat, 50, 65)

# White Blood Cells
reformat = deloutabove(reformat, 52, 500)

# Platelets
reformat = deloutabove(reformat, 53, 2000)

# INR
reformat = deloutabove(reformat, 57, 20)

# pH
reformat = deloutbelow(reformat, 58, 6.7)
reformat = deloutabove(reformat, 58, 8)

# pO2
reformat = deloutabove(reformat, 59, 700)

# pCO2
reformat = deloutabove(reformat, 60, 200)

# Base Excess
reformat = deloutbelow(reformat, 61, -50)

# Lactate
reformat = deloutabove(reformat, 62, 30)

####################################################################
# More data manipulation / imputation from existing values

# Estimate GCS from RASS - data from Wesley JAMA 2003
reformat[(np.isnan(reformat[:, 5])) & (reformat[:, 6] >= 0), 5] = 15
reformat[(np.isnan(reformat[:, 5])) & (reformat[:, 6] == -1), 5] = 14
reformat[(np.isnan(reformat[:, 5])) & (reformat[:, 6] == -2), 5] = 12
reformat[(np.isnan(reformat[:, 5])) & (reformat[:, 6] == -3), 5] = 11
reformat[(np.isnan(reformat[:, 5])) & (reformat[:, 6] == -4), 5] = 6
reformat[(np.isnan(reformat[:, 5])) & (reformat[:, 6] == -5), 5] = 3

# FiO2
reformat[(~np.isnan(reformat[:, 22])) & (np.isnan(reformat[:, 23])), 23] = reformat[(~np.isnan(reformat[:, 22])) & (
    np.isnan(reformat[:, 23])), 22] / 100
reformat[(~np.isnan(reformat[:, 23])) & (np.isnan(reformat[:, 22])), 22] = reformat[(~np.isnan(reformat[:, 23])) & (
    np.isnan(reformat[:, 22])), 23] * 100

print('Full ICU -- Doing sample and hold')
sample_and_hold = pd.read_csv('ReferenceFiles/sample_and_hold.csv', index_col=None)

# reformatsah = SAH(reformat, sample_and_hold)  # Do SAH first to handle this task
reformatsah_physio = SAH(reformat[:, 0:68], sample_and_hold, adjust=0)

reformatsah = np.copy(reformat)
reformatsah[:, 0:68] = reformatsah_physio
# NO FiO2, YES O2 flow, no interface OR cannula
ii = np.where((np.isnan(reformatsah[:, 22])) & (~np.isnan(reformatsah[:, 24])) & (
            (reformatsah[:, 21] == 0) | (reformatsah[:, 21] == 2)))[0]  # As np.where returns a tuple
reformat[ii[reformatsah[ii, 24] <= 15], 22] = 70
reformat[ii[reformatsah[ii, 24] <= 12], 22] = 62
reformat[ii[reformatsah[ii, 24] <= 10], 22] = 55
reformat[ii[reformatsah[ii, 24] <= 8], 22] = 50
reformat[ii[reformatsah[ii, 24] <= 6], 22] = 44
reformat[ii[reformatsah[ii, 24] <= 5], 22] = 40
reformat[ii[reformatsah[ii, 24] <= 4], 22] = 36
reformat[ii[reformatsah[ii, 24] <= 3], 22] = 32
reformat[ii[reformatsah[ii, 24] <= 2], 22] = 28
reformat[ii[reformatsah[ii, 24] <= 1], 22] = 24

# NO FiO2, NO O2 flow, no interface OR cannula
ii = np.where((np.isnan(reformatsah[:, 22])) & np.isnan(reformatsah[:, 24]) & (
            (reformatsah[:, 21] == 0) | (reformatsah[:, 21] == 2)))[
    0]  # no fio2 given and o2flow given, no interface OR cannula
reformat[ii, 22] = 21

# NO FiO2, YES O2 flow, face mask OR.... OR ventilator (assume it's face mask)
ii = np.where((np.isnan(reformatsah[:, 22])) & (~np.isnan(reformatsah[:, 24])) &
              ((reformatsah[:, 21] == 1) | (reformatsah[:, 21] == 3) | (reformatsah[:, 21] == 4) | (
                          reformatsah[:, 21] == 5) | (reformatsah[:, 21] == 6) | (reformatsah[:, 21] == 9) | (
                           reformatsah[:, 21] == 10)))[0]
reformat[ii[reformatsah[ii, 24] <= 15], 22] = 75
reformat[ii[reformatsah[ii, 24] <= 12], 22] = 69
reformat[ii[reformatsah[ii, 24] <= 10], 22] = 66
reformat[ii[reformatsah[ii, 24] <= 8], 22] = 58
reformat[ii[reformatsah[ii, 24] <= 6], 22] = 40
reformat[ii[reformatsah[ii, 24] <= 4], 22] = 36

# NO FiO2, NO O2 flow, face mask OR ....OR ventilator
ii = np.where(np.isnan(reformatsah[:, 22]) & np.isnan(reformatsah[:, 24]) & (
            (reformatsah[:, 21] == 1) | (reformatsah[:, 21] == 3) |
            (reformatsah[:, 21] == 4) | (reformatsah[:, 21] == 5) | (reformatsah[:, 21] == 6) | (
                        reformatsah[:, 21] == 9) | (reformatsah[:, 21] == 10)))[
    0]  # no fio2 given and o2flow given, no interface OR cannula
reformat[ii, 22] = np.nan

# NO FiO2, YES O2 flow, Non rebreather mask
ii = np.where(np.isnan(reformatsah[:, 22]) & (~np.isnan(reformatsah[:, 24])) & (reformatsah[:, 21] == 7))[0]
reformat[ii[reformatsah[ii, 24] >= 10], 22] = 90
reformat[ii[reformatsah[ii, 24] >= 15], 22] = 100
reformat[ii[reformatsah[ii, 24] < 10], 22] = 80
reformat[ii[reformatsah[ii, 24] <= 8], 22] = 70
reformat[ii[reformatsah[ii, 24] <= 6], 22] = 60

# NO FiO2, NO O2 flow, NRM
ii = np.where(np.isnan(reformatsah[:, 22]) & np.isnan(reformatsah[:, 24]) & (reformatsah[:, 21] == 7))[
    0]  # no fio2 given and o2flow given, no interface OR cannula
reformat[ii, 22] = np.nan

# Update FiO2 columns again
ii = (~np.isnan(reformat[:, 22])) & (np.isnan(reformat[:, 23]))
reformat[ii, 23] = reformat[ii, 22] / 100
ii = (~np.isnan(reformat[:, 23])) & (np.isnan(reformat[:, 22]))
reformat[ii, 22] = reformat[ii, 23] * 100

# Blood Pressure
ii = (~np.isnan(reformat[:, 8])) & (~np.isnan(reformat[:, 9])) & np.isnan(reformat[:, 10])
reformat[ii, 10] = (3 * reformat[ii, 9] - reformat[ii, 8]) / 2
ii = (~np.isnan(reformat[:, 8])) & (~np.isnan(reformat[:, 10])) & np.isnan(reformat[:, 9])
reformat[ii, 9] = (reformat[ii, 8] + 2 * reformat[ii, 10]) / 3
ii = (~np.isnan(reformat[:, 9])) & (~np.isnan(reformat[:, 10])) & np.isnan(reformat[:, 8])
reformat[ii, 8] = 3 * reformat[ii, 9] - 2 * reformat[ii, 10]

# Temperature
# Some values recorded in the wrong column
ii = (reformat[:, 14] > 25) & (reformat[:, 14] < 45)  # tempF close to 37deg??!
reformat[ii, 13] = reformat[ii, 14]
reformat[ii, 14] = np.nan
ii = reformat[:, 13] > 70  # tempC > 70, likely recorded in Farenheit
reformat[ii, 14] = reformat[ii, 13]
reformat[ii, 13] = np.nan
ii = (~np.isnan(reformat[:, 13])) & np.isnan(reformat[:, 14])
reformat[ii, 14] = reformat[ii, 13] * 1.8 + 32
ii = (~np.isnan(reformat[:, 14])) & np.isnan(reformat[:, 13])
reformat[ii, 13] = (reformat[ii, 14] - 32) / 1.8

# Hb/Ht
ii = (~np.isnan(reformat[:, 49])) & np.isnan(reformat[:, 50])
reformat[ii, 50] = (reformat[ii, 49] * 2.862) + 1.216
ii = (~np.isnan(reformat[:, 50])) & np.isnan(reformat[:, 49])
reformat[ii, 49] = (reformat[ii, 50] - 1.216) / 2.862

# Bilirubin
ii = (~np.isnan(reformat[:, 43])) & np.isnan(reformat[:, 44])
reformat[ii, 44] = (reformat[ii, 43] * 0.6934) - 0.1752
ii = (~np.isnan(reformat[:, 44])) & np.isnan(reformat[:, 43])
reformat[ii, 43] = (reformat[ii, 44] + 0.1752) / 0.6934

########################################################################
#                      SAMPLE AND HOLD on RAW DATA
########################################################################
print('Full ICU -- SAMPLE AND HOLD on RAW DATA')
reformat_physio1 = reformat[:, 0:68]

reformat[:, 0:68] = SAH(reformat_physio1, sample_and_hold, adjust=0)


for col in [70, 71, 72, 73, 74]:
    reformat[reformat[:, col] < 0, col] = 0


idx_name_map = {
    1: 'icu_id',
    5: 'GCS',
    12: 'SpO2',
    22: 'FiO2',
    44: 'Bilirubin',
    59: 'PaO2',
    62: 'Lactate',
    70: 'Urine_Step',
    71: 'Urine_Total',
    72: 'Vaso_Rate' ,
    73: 'Fluids_Step' ,
    74: 'Fluids_Total'
}

# Select only relevant columns for PINN training
relevant_indices = [i for i in range(68)] + [70, 71, 72, 73, 74]


pinn_df = pd.DataFrame(reformat[:, relevant_indices], columns=relevant_indices)

pinn_df['hours'] = pinn_df.groupby(1)[2].transform(lambda x: (x - x.min()) / 3600.0)

if 22 in pinn_df.columns:
    pinn_df.loc[pinn_df[22] <= 1.0, 22] *= 100
    pinn_df.loc[pinn_df[22] < 21, 22] = 21

pinn_data_unique = pinn_df.groupby([1, 'hours'], as_index=False).mean()

pinn_data_unique.rename(columns=idx_name_map, inplace=True)

final_named_cols = ['icu_id', 'hours', 'SpO2', 'FiO2', 'PaO2', 'Lactate', 'Bilirubin', 'GCS', 'Urine_Step', 'Urine_Total','Vaso_Rate', 'Fluids_Step',  'Fluids_Total']

for col in final_named_cols:
    if col not in pinn_data_unique.columns:
        pinn_data_unique[col] = np.nan

pinn_data_unique_final = pinn_data_unique[final_named_cols].sort_values(by=['icu_id', 'hours'])

_intermediate_csv = 'mimic_pinn_realtime_final.csv'
pinn_data_unique_final.to_csv(_intermediate_csv, index=False)

print(f"max length: {pinn_data_unique_final['hours'].max():.2f} hours")
print(f"vars: {final_named_cols}")

########################################################################
#                         COHORT FILTERING
########################################################################

output_csv = 'mimic_pinn_v4_filtered.csv'

min_valid_vars = 6         # minimum number of non-NaN, non-zero variables per row
min_rows_per_patient = 40  # minimum number of rows per icu_id

data = pd.read_csv(_intermediate_csv)

value_cols = [c for c in data.columns if c not in ['icu_id', 'hours']]
valid_count = ((data[value_cols].notna()) & (data[value_cols] != 0)).sum(axis=1)
data_filtered = data[valid_count >= min_valid_vars]

icu_counts = data_filtered['icu_id'].value_counts()
valid_icu_ids = icu_counts[icu_counts >= min_rows_per_patient].index
data_filtered = data_filtered[data_filtered['icu_id'].isin(valid_icu_ids)]

data_filtered.to_csv(output_csv, index=False)

print(f"Original rows: {len(data)}, filtered rows: {len(data_filtered)}")
print(f"Filtered dataset saved to: {output_csv}")