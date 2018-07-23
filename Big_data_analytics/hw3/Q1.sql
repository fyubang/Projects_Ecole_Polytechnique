with authid_count as 
	(select pa1.authid as ai, count(distinct pa2.authid) as num_co_auth
	from paperauths pa1, paperauths pa2
	where pa1.paperid = pa2.paperid
	and pa1.authid <> pa2.authid
	group by ai
	order by num_co_auth desc
	limit 10)
select authid_count.ai as authid, authors.name as name, authid_count.num_co_auth
from authid_count, authors
where authid_count.ai = authors.id；