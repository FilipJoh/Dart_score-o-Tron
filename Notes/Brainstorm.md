# Dart

Placera ut ett par punkter motsvarande dartpilarnas toppar, och hitta vart de befinner sig i dartboardens koordinatsystem.

Givet listan med punkter, ska vi kunna räkna ut poängen.

Scope:
Givet en bild av darttavlan, ska vi hitta den sammanlagda poängen av dartrundan för en spelare.

Fråga 1:
Identifiera tavlans koordinatsystem, D.v.s hitta den homografi beskriver darttavlans orientering.

Fråga 2:
Hitta dartpilspetsarnas koordinatsystem.

3. Transformera pilarnas koordinater till tavelsystemet.

Sidospår, hitta på några karateristiska punkter för alla darttavlor (svårt)
Viola jones

Vår idé:
Hitta tavlans mittpunkt samt två circlar med ett given relation mellan dem. Sampla från dessa för att skapa homografin.


## Related work:
* Repository:
https://github.com/matherm/matherm.github.io/tree/master/assets/code/dart

* Webarticle: https://matherm.github.io/2018/02/02/the-score-is-nine/

* Paper:
https://web.stanford.edu/class/ee368/Project_Autumn_1516/Reports/Delaney.pdf

* Conic Correspondence:
http://www.macs.hw.ac.uk/bmvc2006/papers/306.pdf

* Conic detection?
http://vision.gel.ulaval.ca/~jouellet/publications/DualConicMVA.pdf
