with paper_coauth as (
	select paperid, count(authid) as num_coauth
	from paperauths
	group by paperid)
select year, num_coauth, count(*)
from venue, papers, paper_coauth
where paper_coauth.paperid = papers.id
and papers.venue = venue.id
and year > 0
group by year, num_coauth
order by year, num_coauth asc;