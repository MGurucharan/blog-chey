import Rating from '@mui/material/Rating';
import Stack from '@mui/material/Stack';

export default function HalfRating() {
  return (
    <Stack spacing={10}>
      <Rating
        name="half-rating"
        defaultValue={0.0}
        precision={0.5}
        sx={{
          color: 'transparent', // Prevent default fill
          '& .MuiRating-icon': {
            color: 'gray',
            stroke: '#aaa',         // Gray stroke color
            strokeWidth: 0.1,       // Thicker border
            fontSize: '1.5rem',       // Optional: Increase size
          },
          '& .MuiRating-iconFilled': {
            color: '#ffc107',       // Fill color for active stars
            stroke: 'transparent',  // No border when filled
          },
          '& .MuiRating-iconHover': {
            color: '#ffc107',       // Hover fill color
            stroke: 'transparent',
          },
        }}
      />
    </Stack>
  );
}
