with authid_count as 
	(select pa1.authid as ai, count(distinct pa2.authid) as num_co_auth
	from paperauths pa1, paperauths pa2
	where pa1.paperid = pa2.paperid
	and pa1.authid <> pa2.authid
	group by ai
	order by num_co_auth desc)
select num_co_auth, count(distinct ai) as num_auth
from authid_count
group by num_co_auth
order by num_co_auth;