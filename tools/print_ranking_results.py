from bs4 import BeautifulSoup

def convert_HTML(text):
    # remove duplicate text + extract code blocks 
    # return pure text and code blocks
    
    bs_text = BeautifulSoup(text)
    text = "Possible Duplicate:"
    dup_can = bs_text.find(lambda tag: tag.name == "strong" and text in tag.text)
    if dup_can!=None:
        wrapping = dup_can.find_previous('blockquote')
        if wrapping!= None: wrapping.clear()
#         print(dup_can.parent.parent)

    text = "This is a duplicate of"
    dup_can = bs_text.find(lambda tag: tag.name == "p" and text in tag.text)
    if dup_can!=None:
        wrapping = dup_can.clear()

    # extract code block
    
    cb_list = []
#     cbs = bs_text.find_all("code")
    cbs = bs_text.select("pre > code")
    
    if len(cbs)!=0:
        for cb in cbs:
            cb_list.append(cb.get_text())
            cb.clear()
            
    text = bs_text.get_text()
    return text, cb_list

def print_post(post):
    print('Id:',post['Id'])
    print('Title:',post['Title']+'\n')
    print('Body:',convert_HTML(post['Body'])[0])
    print('-----------------------------------')

def return_pred_top_n(i, testset, label_list, ranking_list, top_n = 3):

    list_key = list(testset['dup'].keys())
    query = testset['dup'][list_key[i]]
    print('Query:')
    print_post(query)
    gt = testset['rel'][list_key[i]]
    
    rl = ranking_list[i].numpy().astype(int)
    for i_g, g in enumerate(gt):
        cur_gt = testset['ori'][g]

        ind_pred = np.where(rl==g)
        if len(ind_pred)!=0:
            print('GT ranking: ',(ind_pred[0]))
        else:
            print('Not in list.')
        
        print('ground truth %d:'%(i_g))
        print_post(cur_gt)
    
    for i_n in range(top_n):
        print('Candidate %d:'%i_n)
        print_post(testset['ori'][int(ranking_list[i][i_n].item())])


return_pred_top_n(14, test_set, label_list, ranking_list)