#Copyright 2015 Arun Kumar
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import sys,math,numpy
import matplotlib.pyplot as plt

drs = [2, 4, 20, 40, 200, 400]
nss = [10000, 50000, 100000, 500000, 1000000, 5000000]
mdr = 40
mns = 500000
mds = 4
mnr = 10000

jointimes = {} #indexed by paramcombo
avgall = {} #indexed by FL:paramcombo
cntall = {} #indexed by FL:paramcombo
sumsqall = {} #to plot CIs 
stdevall = {} #to plot CIs

filname = 'datasynth.out'
filh = open(filname)
v = ""
for line in filh :
	l = [i.strip() for i in line.split(' ')]
	v2 = ""
	if (l[0] == "GEN_DATA:") :
		v = ":".join([l[1], l[2], l[3], l[4]])
		cnt = 0
	if (l[0] == "TIME_J:") :
		jointimes[v] = l[3]
	if ((l[0] == "TIME_SL:") or (l[0] == "TIME_SS:") or (l[0] == "TIME_FL:") or (l[0] == "TIME_FS:")) :
		alg = [i.split(':') for i in l[0].split('_')][1][0]
		v2 = ":".join([alg, v])
		if(avgall.has_key(v2)) :
			avgall[v2] += 1.0 * float(l[3])
			sumsqall[v2] += 1.0 * float(l[3]) * float(l[3])
			cntall[v2] += 1
		else :
			avgall[v2] = 1.0 * float(l[3])
			sumsqall[v2] = 1.0 * float(l[3]) * float(l[3])
			cntall[v2] = 1

for k, v in avgall.iteritems():
	avgall[k] /= cntall[k]
	stdevall[k] = math.sqrt(sumsqall[k] / cntall[k] - avgall[k] * avgall[k])
	#print 'k:', k, 'cnt:', cntall[k], 'avg:', avgall[k], 'sumsq:', sumsqall[k], 'stdev:', stdevall[k]

legptsl = {'Den. Learning':['SL', '-o','red'], 'Fact. Learning':['FL', '--','blue']}
legptss = {'Den. Scoring':['SS', '-o','red'], 'Fact. Scoring':['FS', '--','blue']}
plt.clf()
font = {'size': 22}
plt.rc('font', **font)

fig = plt.figure(figsize=(8, 8))
plt.ylabel("Runtime (s) in logscale")
plt.xlabel("Number of Tuples in S (nS) in logscale")
#if((xk == 'nr') or (xk == 'nl')) :
plt.xscale('log')
#plt.yscale('log')
for lek, lev in legptsl.iteritems() :
	#print lek, lev[0], lev[1], lev[2]
	#print nss, [':'.join([lev[0], str(i), str(mnr), str(mds), str(mdr)]) for i in nss],  [avgall[':'.join([lev[0], str(i), str(mnr), str(mds), str(mdr)])] for i in nss]
	plt.errorbar(nss, [avgall[':'.join([lev[0], str(i), str(mnr), str(mds), str(mdr)])] for i in nss], fmt=lev[1], yerr = [1.645 * a / math.sqrt(b) for a, b in zip([stdevall[':'.join([lev[0], str(i), str(mnr), str(mds), str(mdr)])] for i in nss], [cntall[':'.join([lev[0], str(i), str(mnr), str(mds), str(mdr)])] for i in nss])], color=lev[2])
plt.ylim(ymin=0)
plt.legend(tuple(legptsl.keys()), loc='best')
plt.savefig('Vary-nS-SLFL.svg')
plt.clf()

fig = plt.figure(figsize=(8, 8))
plt.ylabel("Runtime (s) in logscale")
plt.xlabel("Number of Tuples in S (nS) in logscale")
#if((xk == 'nr') or (xk == 'nl')) :
plt.xscale('log')
#plt.yscale('log')
for lek, lev in legptss.iteritems() :
	plt.errorbar(nss, [avgall[':'.join([lev[0], str(i), str(mnr), str(mds), str(mdr)])] for i in nss], fmt=lev[1], yerr = [1.645 * a / math.sqrt(b) for a, b in zip([stdevall[':'.join([lev[0], str(i), str(mnr), str(mds), str(mdr)])] for i in nss], [cntall[':'.join([lev[0], str(i), str(mnr), str(mds), str(mdr)])] for i in nss])], color=lev[2])
plt.ylim(ymin=0)
plt.legend(tuple(legptsl.keys()), loc='best')
plt.savefig('Vary-nS-SSFS.svg')
plt.clf()

fig = plt.figure(figsize=(8, 8))
plt.ylabel("Runtime (s) in logscale")
plt.xlabel("Number of features in R (dR) in logscale")
#if((xk == 'nr') or (xk == 'nl')) :
plt.xscale('log')
#plt.yscale('log')
for lek, lev in legptsl.iteritems() :
	plt.errorbar(drs, [avgall[':'.join([lev[0], str(mns), str(mnr), str(mds), str(i)])] for i in drs], fmt=lev[1], yerr = [1.645 * a / math.sqrt(b) for a, b in zip([stdevall[':'.join([lev[0], str(mns), str(mnr), str(mds), str(i)])] for i in drs], [cntall[':'.join([lev[0], str(mns), str(mnr), str(mds), str(i)])] for i in drs])], color=lev[2])
plt.ylim(ymin=0)
plt.legend(tuple(legptsl.keys()), loc='best')
plt.savefig('Vary-dR-SLFL.svg')
plt.clf()

fig = plt.figure(figsize=(8, 8))
plt.ylabel("Runtime (s) in logscale")
plt.xlabel("Number of features in R (dR) in logscale")
#if((xk == 'nr') or (xk == 'nl')) :
plt.xscale('log')
#plt.yscale('log')
for lek, lev in legptss.iteritems() :
	plt.errorbar(drs, [avgall[':'.join([lev[0], str(mns), str(mnr), str(mds), str(i)])] for i in drs], fmt=lev[1], yerr = [1.645 * a / math.sqrt(b) for a, b in zip([stdevall[':'.join([lev[0], str(mns), str(mnr), str(mds), str(i)])] for i in drs], [cntall[':'.join([lev[0], str(mns), str(mnr), str(mds), str(i)])] for i in drs])], color=lev[2])
plt.ylim(ymin=0)
plt.legend(tuple(legptss.keys()), loc='best')
plt.savefig('Vary-dR-SSFS.svg')
plt.clf()
