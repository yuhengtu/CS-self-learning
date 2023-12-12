#include <stddef.h>
#include "ll_cycle.h"

/*
typedef struct node {
	    int value;
	    struct node *next;
} node;
*/

int check(node *h, node *t)
{
	if(!h) return 0;
	h = h->next;
	if(!h) return 0;
	h = h->next;
	if(!h) return 0;
	t = t->next;
	if(t == h) return 1;
	else return check(h, t);
}

int ll_has_cycle(node *head) 
{
    return check(head, head);
}
