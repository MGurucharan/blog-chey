
import { useState } from "react"
import { NavLink } from "react-router-dom"

const Navbar = () => {
  const [isOpen, setOpen] = useState(false)

  const toggleNavbar = () => {
    setOpen(!isOpen)
  }

  return (
    <nav className="p-5 shrink-0 w-full fixed z-50 top-4 left-1/2 transform -translate-x-1/2 max-w-6xl">
      {/* Top Bar */}
      <div className="bg-white/5 backdrop-blur-md border border-white/20 rounded-full px-6 py-3 shadow-lg">
        {/* Logo */}
        <div className="flex justify-between items-center h-10 overflow-visible">
        <div className="shrink-0 ml-4 md:ml-10">
          <NavLink to="/" className="focus:outline-none">
            <img
              className="font-semibold object-contain w-35 h-35"
              src="/blogchey.png"
              alt="logo"
            />
          </NavLink>
        </div>

        {/* Desktop Menu */}
        <div className="hidden md:flex items-center gap-7 mr-10">
          <NavLink
            to="/create"
            className="text-white text-[20px] font-semibold hover:text-amber-300 transition duration-300"
          >
            Create
          </NavLink>
          <NavLink
            to="/profile"
            className="text-white text-[20px] font-semibold hover:text-amber-300 transition duration-300"
          >
            Profile
          </NavLink>
          <NavLink
            to="/login"
            className="text-white text-[20px] font-semibold border-2 border-amber-300/50 rounded-full px-4 py-1 hover:text-amber-300 hover:border-amber-300 hover:bg-amber-300/10 transition duration-300"
          >
            Login
          </NavLink>
          <NavLink
            to="/signup"
            className="whitespace-nowrap text-white text-[20px] font-semibold border-2 border-amber-300/50 rounded-full px-4 py-1 hover:text-amber-300 hover:border-amber-300 hover:bg-amber-300/10 transition duration-300"
          >
            Sign up
          </NavLink>
        </div>

        {/* Hamburger Button */}
        <button
          onClick={toggleNavbar}
          className="shrink-0 block mr-4 md:hidden focus:outline-none"
          aria-label="Toggle menu"
        >
          <img
            className="cursor-pointer w-6 h-6"
            src="/more.png"
            alt="Menu"
          />
        </button>
      </div>
    </div>
      {/* Mobile Dropdown (separate panel below navbar) */}
      <div
        className={`md:hidden transition-all duration-300 ease-in-out ${
          isOpen ? "max-h-[400px] opacity-100 mt-3" : "max-h-0 opacity-0 overflow-hidden"
        }`}
      >
        <div className="flex flex-col gap-4 bg-black/80 backdrop-blur-lg px-6 py-6 rounded-xl border border-white/10 shadow-lg">
          <NavLink
            to="/create"
            className="text-white text-[18px] hover:text-amber-300 transition duration-300"
            onClick={() => setOpen(false)}
          >
            Create
          </NavLink>
          <NavLink
            to="/profile"
            className="text-white text-[18px] hover:text-amber-300 transition duration-300"
            onClick={() => setOpen(false)}
          >
            Profile
          </NavLink>
          <div className="flex gap-4 items-center">
            <NavLink
              to="/login"
              className="text-white text-[18px] border-2 border-amber-300/50 rounded-full px-3 py-1 hover:text-amber-300 hover:border-amber-300 hover:bg-amber-300/10 transition duration-300"
              onClick={() => setOpen(false)}
            >
              Login
            </NavLink>
            <NavLink
              to="/signup"
              className="text-white text-[18px] border-2 border-amber-300/50 rounded-full px-3 py-1 hover:text-amber-300 hover:border-amber-300 hover:bg-amber-300/10 transition duration-300"
              onClick={() => setOpen(false)}
            >
              Sign up
            </NavLink>
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
