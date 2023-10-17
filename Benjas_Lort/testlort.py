import numpy as np

Eng    = np.array([[42, 193, 148],[44, 205, 152],[41, 210, 141],[40, 197, 149],[40, 220, 138],[42, 214, 149],[37, 183, 128],[40, 219, 134],[40, 189, 119]])
Mark   = np.array([[26, 233, 189],[26, 233, 193],[26, 243, 193],[26, 233, 189],[26, 241, 186],[26, 236, 174],[27, 216, 168],[27, 243, 185],[26, 240, 186]])
Skov   = np.array([[42, 176, 68],[40, 166, 63],[40, 175, 64],[42, 166, 63],[40, 174, 63],[38, 162, 63],[40, 111, 71],[36, 181, 62],[34, 153, 70]])
Vand   = np.array([[104, 206, 129],[105, 233, 138],[106, 219, 129],[107, 236, 137],[106, 240, 153],[106, 236, 144],[105, 242, 160],[104, 233, 161],[105, 191, 140]])
Vissen = np.array([[23, 124, 111],[23, 130, 114],[23, 160, 113],[26, 250, 165],[23, 130,  94],[21, 156,  93],[22, 131, 121],[23, 158, 121],[25, 118, 115]])
Mine   = np.array([[22, 130,  51],[23, 125,  51],[23, 118,  56],[24, 134,  63],[20, 115,  40],[24, 118,  52],[23, 190,  55],[23, 157,  39],[22, 147,  40]])



np.reshape(Eng,    (9,3))
np.reshape(Mark,   (9,3))
np.reshape(Skov,   (9,3))
np.reshape(Vand,   (9,3))
np.reshape(Vissen, (9,3))
np.reshape(Mine,   (9,3))
print(Vand)
Eng_Tresh    = np.array([[np.min(Eng, axis=0)],    [np.max(Eng, axis=0)]])
Mark_Tresh   = np.array([[np.min(Mark, axis=0)],   [np.max(Mark, axis=0)]])
Skov_Tresh   = np.array([[np.min(Skov, axis=0)],   [np.max(Skov, axis=0)]])
Vand_Tresh   = np.array([[np.min(Vand, axis=0)],   [np.max(Vand, axis=0)]])
Vissen_Tresh = np.array([[np.min(Vissen, axis=0)], [np.max(Vissen, axis=0)]])
Mine_Tresh   = np.array([[np.min(Mine, axis=0)],   [np.max(Mine, axis=0)]])
print("Eng", Vand_Tresh)
#print("Mark",Mark_Tresh)
#print("Skov",Skov_Tresh)
#print("Vand",Vand_Tresh)
#print("Vissen",Vissen_Tresh)
#print("Mine",Mine_Tresh)


#print(np.max(Eng, axis=0))
#print(np.min(Eng, axis=0))




