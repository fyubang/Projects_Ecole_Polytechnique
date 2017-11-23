with num_auth as (
	select paperid, count(distinct authid) as num_auth
	from paperauths
	group by paperid 
	), avg_auth as (
	select p.venue as venue_id, avg(n.num_auth) as avg_num_auth
	from papers p, num_auth n
	where p.id = n.paperid
	group by p.venue
	order by avg_num_auth desc
	limit 5)
select v.id, v.name, a.avg_num_auth
from venue v, avg_auth a
where v.id = a.venue_id;