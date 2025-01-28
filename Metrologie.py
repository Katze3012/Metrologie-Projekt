'''
    Programm zur Durchführung einer Monte-Carlo-Simulation um die Flugzeit eines unbekannten Isotps zu berechnen.
    Als Startwerte sind die Flugzeiten und deren Unsicherheiten von jeweils Bor und Silber gegeben.
    Um das Programm ausführen zu können wird die neuste Version von Phython benötigt und jeweils die Phyton Bibliotheken scipy,numpy und matplotlib.
    Falls nicht vorhanden können diese mit pip install installiert werden.
'''
import numpy as np 
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.stats
#Gegebene Werte zur Berechnung der Ergebnisse

sims = 10000                            #Stichprobenanzahl
A_B = 10                                #Massenzahl Bor untere Grenze
mu_B = 6.1                              #Messwert Bor
sigma_B = 0.5                           #Unsicherheit Bor
m_B = 1/(np.sqrt(2*np.pi)*sigma_B)      #Majorisierungsfaktor für Rejection Sampling von Bor bzw. Maximum von Normalverteilung
A_Ag= 109                               #Massenzahl Silber obere Grenze
mu_Ag = 12.2                            #Messwert Silber
sigma_Ag = 1.2                          #Unsicherheit Silber
m_Ag = 1/(np.sqrt(2*np.pi)*sigma_Ag)    #Majorisierungsfaktor für Rejection Sampling von Silber bzw. Maximum von Normalverteilung
A = 48                                  #Massenzahl des unbekannten Atoms/Isotops
m = 61/990                              #Steigung von Gerade

#Ergebnisse der Zufallsziehungen
random_B = []           #Array für die zufällige gezogenen TOF von Bor 
random_Ag = []          #Array für die zufällige gezogenen TOF von Silber 
random_result = []      #Array für die Ergebnisse von A=48

#Normierungskonstanten; Berechnung nach der Formel gegeben im Paper
c_B = scipy.integrate.quad(lambda x: m_B*np.exp(-(x-mu_B)**2/(2*sigma_B**2)),-3*sigma_B+mu_B, 3*sigma_B+mu_B)          #Normierungskonstante für Bor
c_Ag = scipy.integrate.quad(lambda x: m_Ag*np.exp(-(x-mu_Ag)**2/(2*sigma_Ag**2)),-3*sigma_Ag+mu_Ag, 3*sigma_Ag+mu_Ag)  #Normierungskonstante für Silber

#Funktionen generieren eine Zufallszahl für Bor und Silber nach den Schritten 1 bis 5 in der Anleitung https://ikpcloud.ikp.physik.tu-darmstadt.de/index.php/s/w92HZR5VPFGeEwt?path=%2FProjektgruppe_C_Programmieren%2FC_Kurvenanpassung_Monte-Carlo#pdfviewer

#Berechnung von Zufallszahlen von Bor
def random_var_boron():
    while True:                                                                                 #while Schleife um Rejection Sampling zu erlauben
        x = scipy.stats.uniform.rvs(loc= -3*sigma_B+mu_B,scale = 6*sigma_B+mu_B)                #Ziehung einer Zufallszahl x aus einer Rechtecksverteilung um 0 mit breite -3sigma+mu bis 3sigma+mu; siehe Dokumentation von scipy für loc und scale
        B_norm = scipy.stats.norm.pdf(x,mu_B,sigma_B)/c_B[0]                                    #Berechnung vom dem normalverteilten Wert dividiert durch die Normierungskonstante für die oben gezogene rechteckverteilte Zufallszahl
        r = scipy.stats.uniform.rvs(loc= 0,scale = m_B)                                         #Ziehung einer Zufallszahl aus einer Rechtecksverteilung in einem Intervall [0,m), wobei m eine Funktionsabhängige Majorisierungkonstante ist; Hier Hochpunkt der Normalverteilung
        if r < B_norm:                                                                          #Eigentlicher Vergleich des Rejection Samplings; damit der Wert erlaubt wird muss f(x)=B_norm>r sein, also anschaulich unter dem Graphen liegen
            return x                                                                            #Rückgabe des angenohmenen x-Wertes

#Berechnung der Zufallszahlen von Silber; der Code ist fast analog zu dem von Bor bis auf die jeweiligen Atom bezogenen Konstanten
def random_var_silver():
    while True:
        x = scipy.stats.uniform.rvs(loc= -3*sigma_Ag+mu_Ag,scale = 6*sigma_Ag+mu_Ag)
        Ag_norm = scipy.stats.norm.pdf(x,mu_Ag,sigma_Ag)/c_Ag[0]
        r = scipy.stats.uniform.rvs(loc= 0,scale = m_Ag)
        if r < Ag_norm:
            return x

#Eigentliche Monte-Carlo-Simulation
for i in range(sims):                                                               #for-Schleife um für jede Stichprobenzahl einen Durchlauf zu erlauben
    random_B += [random_var_boron()]                                                #Jeder errechnete Wert für Bor wird in einem Array gespeichert
    random_Ag += [random_var_silver()]                                              #Analog zu Bor        
    random_result += [((random_Ag[i-1]-random_B[i-1])/99)*(A-10)+random_B[i-1]]     #Berechnung des zufälligen Erbenisses für A=48 aus den Werten von Bor und Silber; Array Zugriffe haben Eintrag i-1 wegen Phyton internen Definitionen 

#Berechungen der Mittelwerte und Standardabweichungen von jeweils Bor und A=48
boron_mean = np.mean(random_B)
boron_std = np.std(random_B)
result_mean = np.mean(random_result)
result_std = np.std(random_result)

#Histogram des Ergebnisses für die Zufallswerte von Bor
plt.hist(random_B, bins=100)                                                            #Eigentlichen Histogramm
plt.suptitle('Verteilung von TOF von Bor')                                              #Titel des Histogramms
plt.axvline(boron_mean, color = 'y')                                                    #Gelbe Linie beim Mittelwert
plt.axvline(boron_mean+boron_std, color = 'tab:orange')                                 #Orangene Linie bei mu+sigma
plt.axvline(boron_mean-boron_std, color = 'tab:orange')                                 #Orangene Linie bei mu-sigma
plt.text(boron_mean,260,"  x̂ = " + format(boron_mean),color= 'y')                       #Gelber Text beim Mittelwert
plt.text(boron_std+boron_mean,250,"  u(x̂) = " + format(boron_std),color= 'tab:orange')  #Orangener Text bei der rechten mu+sigma Linie
plt.xlabel('Zeit in ns')                                                                #x-Achsen Beschriftung
plt.show()                                                                              #Anzeigen des Histogramms

#Histogram des Ergebnisses für A=48; selbige Kommentare wie bei Bor erklären den Code
plt.hist(random_result, bins=100)
plt.suptitle('Verteilung von TOF A=48')
plt.axvline(result_mean, color = 'y')
plt.axvline(result_mean+result_std, color = 'tab:orange')
plt.axvline(result_mean-result_std, color = 'tab:orange')
plt.xlabel('Zeit in ns')
plt.text(result_mean,260,"  x̂ = " + format(result_mean),color= 'y')
plt.text(result_std+result_mean,250,"  u(x̂) = " + format(result_std),color= 'tab:orange')
plt.show()