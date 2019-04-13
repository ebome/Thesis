'''
HWAM in USYD calculator
'''

# ye2=input('enter 2nd yr subject scores, not incld X111:')
# year2 = eval(ye2)
year2 =  [55,76,80,72,71,83,60,73]

# ye3=input('enter 3rd yr subject scores, not incld X111:') # input as a string, but cast into list
# year3 = eval(ye3)
year3 =   [73,80,86,75,76,85,84]

# ye4=input('enter 4th yr subject scores, not incld X111:')
# year4 = eval(ye4)
year4  =  [87,82,83]

thesis=input('enter thesis scores:')
thesis_score = int(thesis)


accumulation=0
for yr2 in year2:
    accumulation = accumulation + 2*6*yr2
for yr3 in year3:
    accumulation = accumulation + 3*6*yr3
for yr4 in year4:
    accumulation = accumulation + 4*6*yr4

# Add two-credit ENGGX111 into total score
enggX111 = 2 * (2*78 + 3*73 + 4*78)
thesis = thesis_score*4*6*2
accumulation = accumulation +  enggX111 + thesis
    
denominator = 2*6*len(year2)+3*6*len(year3)+4*6*len(year4)+2*(2 + 3 + 4)  + 8*6

HWMA = accumulation/denominator

print('your current hwam is: ' + str(HWMA))

# https://www.jianshu.com/p/0424d49f325d



